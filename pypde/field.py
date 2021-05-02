import numpy as np
from .bases.spectralbase import SpectralBase
from .bases.memoize import memoized

class SpectralSpace():
    '''
    Class that handles all spectral bases and generalizes
    them to multidimensional spaces

    Input
        shape: int tuple (ndim)
            Shape of field in real space, can be 1d or 2d
        bases: str tuple
            Define the spectral bases with a key
            ('CH','CD','CN','FO')
            See bases.SpectralSpace
    '''
    def __init__(self,shape,bases):
        shape,bases = self._check_input(shape,bases)
        assert len(shape) == len(bases), "Shape size must match number of bases"
        self.shape_physical = shape
        self.ndim = len(self.shape_physical)
        self._set_bases(bases)

    def _check_input(self,shape,bases):
        if isinstance(shape, int): shape = (shape,)
        if isinstance(bases, str): bases = (bases,)
        return shape, bases

    def _set_bases(self,bases):
        self.xs = []
        shape_spectral = []
        for i,key in enumerate(bases):
            N = self.shape_physical[i]
            self.xs.append(SpectralBase(N,key))
            shape_spectral.append(self.xs[i].M)
        self.shape_spectral = tuple(shape_spectral)

    def forward_fft(self,v,axis,bc=None):
        assert isinstance(axis,int)
        #for axis,x in enumerate(self.xs):
        if axis == 0:
            vhat = self.xs[axis].forward_fft(v,bc=bc)
        else:
            vhat = np.swapaxes(v,axis,0)
            bcc = np.swapaxes(bc,axis,0) if bc is not None else None
            vhat = self.xs[axis].forward_fft(vhat,bc=bcc)
            vhat = np.swapaxes(vhat,axis,0)
        return vhat

    def backward_fft(self,vhat,axis,bc=None):
        assert isinstance(axis,int)
        if axis == 0:
            v = self.xs[axis].backward_fft(vhat,bc=bc)
        else:
            v = np.swapaxes(vhat,axis,0)
            bcc = np.swapaxes(bc,axis,0) if bc is not None else None
            v = self.xs[axis].backward_fft(v,bc=bcc)
            v = np.swapaxes(v,axis,0)
        return v

class SpectralSpaceBC(SpectralSpace):
    '''
    Handles a boundary conditions along 1 axis
    This class is used internally.

    Input
        shape: tuple
            If boundary condition is applied,
            say at x=0 this the shape is [2,ny].
        bases: tuple
            See parent
        axis: int
            Axis along which bc is applied
        value: array (shape)
            Boundary condition in physical space
    '''
    def __init__(self,shape,bases,axis,value):
        if shape[axis] != 2:
            raise ValueError("Size of boundary condition along axis must be 2!")
        SpectralSpace.__init__(self,shape,bases)
        for b in self.xs:
            assert b.bc is not None
        self.axis = axis
        self._value = value

    def forward_fft(self,v,axis):
        '''
        Since the BC functions are defined with +1 or -1 endpoints,
        the coefficients bc of the first transform are equal
        to the bc values in physical space.
        However, when transforming further (2d), the coefficients
        must be transformed too, along all other axis
        '''
        assert isinstance(axis,int)
        assert v.shape[self.axis] == 2
        if axis == self.axis:
            return v
        elif axis == 0:
            return self.xs[axis].forward_fft(v)
        else:
            vhat = np.swapaxes(v,axis,0)
            vhat = self.xs[axis].forward_fft(vhat)
            vhat = np.swapaxes(vhat,axis,0)
            return vhat

    def backward_fft(self):
        ''' Not necessary for bc coefficients '''
        raise NotImplementedError

    @property
    @memoized
    def value(self):
        return self._value

    @property
    @memoized
    def value_for_forward(self):
        '''
        BCs have to be transformed for axis: 0 --> self.axis

        This is only true if forward transforms are performed in
        ascending order, i.e. from axis=0 to ndim
        '''
        vhat = self.value
        for axis in range(self.axis):
            vhat = self.forward_fft(vhat,axis=axis)
        return vhat

    @property
    @memoized
    def value_for_backward(self):
        '''
        BCs have to be transformed for axis: ndim --> self.axis

        This is only true if backward transforms are performed in
        ascending order, i.e. from axis=0 to ndim
        '''
        vhat = self.value
        for axis in np.arange(self.ndim-1,self.axis,-1):
            vhat = self.forward_fft(vhat,axis=int(axis))
        return vhat


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
    > bcy = np.zeros((N,2))
    > bcy[:,0] = field.x
    > field.add_bc(bcx,axis=0)
    > field.add_bc(bcy,axis=1)
    >
    > # Transform
    > f = np.sin(np.pi* xx)
    > fhat = field.forward(f)
    > fcd = field.backward(fhat)

    '''
    def __init__(self,shape,bases):
        SpectralSpace.__init__(self,shape,bases)
        self.bases = bases
        # Field in physical space
        self.v = np.zeros(self.shape_physical)
         # Field in spectral space
        self.vhat = np.zeros(self.shape_spectral)

        # Boundary Conditions
        self.bcs = {}

    @property
    def x(self):
        return self.xs[0].x

    @property
    def y(self):
        if self.ndim<2:
            raise ValueError("Dimension y not defined for ndim<2.")
        return self.xs[1].x

    def forward(self,v=None,with_bcs=True):
        '''
        Full forward transform
        '''
        assert v.shape == self.shape_physical
        if v is None: v = self.v
        vhat = v

        # -- Add boundary conditions
        bc = [None]*self.ndim
        if with_bcs:
            for axis in self.bcs:
                bc[axis] = self.bcs[axis].value_for_forward

        # -- Transform forward
        for axis in range(self.ndim):
            vhat = self.forward_fft(vhat,axis=axis,bc=bc[axis])

        return vhat

    def backward(self,vhat=None,with_bcs=True):
        '''
        Full backward transform
        '''
        assert vhat.shape == self.shape_spectral
        if vhat is None: vhat = self.vhat
        v = vhat

        # -- Add boundary conditions
        bc = [None]*self.ndim
        if with_bcs:
            for axis in self.bcs:
                bc[axis] = self.bcs[axis].value_for_backward

        # -- Transform backward
        for axis in range(self.ndim):
            v = self.backward_fft(v,axis=axis,bc=bc[axis])

        return v

    def add_bc(self,value,axis):
        sbc = list(self.shape_physical); sbc[axis] = 2; sbc = tuple(sbc)
        if value.shape != sbc:
            raise ValueError ("""BC value must be of ndim 2 along axis
            and match the field shape in all other dimensions""")

        key = axis
        if key not in self.bcs:
            self.bcs[key] = SpectralSpaceBC(sbc,self.bases,axis=axis,value=value)
        else:
            raise ValueError("Boundary condition in axis {:} already present!"
            .format(key))
