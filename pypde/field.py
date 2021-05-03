import numpy as np
from .bases.spectralspace import *
from .bases.memoize import memoized

class Field(SpectralSpace):
    '''
    Class for (multidimensional) Fields in
    real and spectral space

    Input
        shape: tuple
            See SpectralSpace
        bases: tuple
            See SpectralSpace

    Methods
        self.forward(f)
            Perform full forward transform of a given input

        self.backward(fhat)
            Perform full forward transform of a given input

        self.add_bc(value,axis)
            Add boundary conditions
            Input
                value: nd array
                    Dimension along axis must be 2
                axis: int
                    Specify axis of boundary condition

    Example
    > from pypde.field import *
    > N,M = 20,10
    > field = Field((N,M),("CD","CD"))
    >
    > # Spatial info
    > xx,yy = np.meshgrid(field.x,field.y,indexing="ij")
    >
    > # Boundary conditions
    > bcx = np.zeros((2,M))
    > bcx[0] = field.y
    > field.add_bc(bcx,axis=0)

    '''
    def __init__(self,shape,bases):
        SpectralSpace.__init__(self,shape,bases)
        self.bases = bases
        # Field in physical space
        self.v = np.zeros(self.shape_physical)
         # Field in spectral space
        self.vhat = np.zeros(self.shape_spectral)

        # Inhomogeneous field
        self.field_bc = None

    @property
    def x(self):
        return self.xs[0].x

    @property
    def y(self):
        if self.ndim<2:
            raise ValueError("Dimension y not defined for ndim<2.")
        return self.xs[1].x

    def forward(self,v=None):
        '''
        Full forward transform to homogeneous field v
        '''
        if v is None: v = self.v
        assert v.shape == self.shape_physical

        vhat = v
        # -- Transform forward
        for axis in range(self.ndim):
            vhat = self.forward_fft(vhat,axis=axis)

        return vhat

    def backward(self,vhat=None):
        '''
        Full backward transform to homogeneous field v
        '''
        if vhat is None: vhat = self.vhat
        assert vhat.shape == self.shape_spectral
        
        v = vhat
        # -- Transform backward
        for axis in range(self.ndim):
            v = self.backward_fft(v,axis=axis)

        return v

    def add_bc(self,bchat,axis):
        import warnings

        if bchat.shape[axis] != 2:
            raise ValueError("BC must be of size 2 along specified axis")
    
        if axis != 0:
            warnings.warn("Boundary Conditions might not work for axis !=0")

        self.field_bc    = FieldBC(self.shape_physical,self.bases,axis)
        self.field_bc.vhat = bchat

    def make_homogeneous(self,v=None):
        '''
        Subtract inhomogeneous field  (in self.field_bc) from v 
        '''
        import warnings
        if self.field_bc is None:
            warnings.warn("No inhomogeneous field found. Call add_bc(bchat,axis) first!")
        else:
            assert v.shape == self.inhomogeneous.shape, \
            "Shape mismatch in make_homogeneous"
        return v - self.inhomogeneous

    @property
    def total(self):
        return self.homogeneous + self.inhomogeneous

    
    @property
    def homogeneous(self):
        return self.v

    @property
    def inhomogeneous(self):
        if self.field_bc is not None:
            return self.field_bc.v
        return 0

class FieldBC(SpectralSpaceBC):
    '''
    Handle inhomogeneous field from inhomogeneous boundary
    conditions. 
    Used in class Field, after calling method add_bc
    '''
    def __init__(self,shape,bases,axis):
        SpectralSpaceBC.__init__(self,shape,bases,axis)
        self.bases = bases
        # Field in physical space
        self._v = np.zeros(self.shape_physical)
         # Field in spectral space
        self._vhat = np.zeros(self.shape_spectral)

    @property
    def v(self):
        return self._v
     
    @v.setter
    def v(self,value):
        self._vhat = self.forward_fft(value,self.axis)
        self._v = value
    
    @property
    def vhat(self):
        return self._vhat

    @vhat.setter
    def vhat(self,value):
        self._v = self.backward_fft(value,self.axis)
        self._vhat = value