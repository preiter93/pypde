import numpy as np
from .bases.spectralspace import *
from .bases.memoize import memoized
from .plot.anim import animate_line,animate_contour,animate_wireframe

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
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    from pypde.field import *
    N,M = 40,30
    shape = (N,M)
    field = Field(shape,("CD","CN"))

    # Spatial info
    xx,yy = np.meshgrid(field.x,field.y,indexing="ij")
    f = np.sin(np.pi* xx)+xx+np.sin(4*yy)
    field.v = f

    # Boundary conditions
    bc = np.zeros((2,M))
    bc[0,:] = -1+np.sin(4*field.y)
    bc[1,:] =  1+np.sin(4*field.y)
    field_bc = FieldBC(shape,("CD","CN"),axis=0)
    field_bc.add_bc(bc)

    # Extract Homogeneous part of f
    field.add_field_bc(field_bc)
    field.v = field.make_homogeneous()

    # Transform
    field.forward()
    field.backward()

    # Plot
    from pypde.plot.wireframe import plot 
    plot(xx,yy,f)
    plot(xx,yy,field.inhomogeneous)
    plot(xx,yy,field.homogeneous)
    plot(xx,yy,field.total)

    # Derivative along x
    field_deriv = derivative_field(field,deriv=(1,0))
    plot(xx,yy,field_deriv.v)

    # Derivative along y (almost zero)
    field_deriv = derivative_field(field,deriv=(0,1))
    plot(xx,yy,field_deriv.v)
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    def __init__(self,shape,bases):
        SpectralSpace.__init__(self,shape,bases)
        self.bases = bases

        self.v = np.zeros(self.shape_physical)      # physical field
        self.vhat = np.zeros(self.shape_spectral)   # spectral field

        self.field_bc = None #inhomogeneous field

        self.t = 0      # time
        self.V = []     # Storage Field
        self.T = []     # Storage Time

    @property
    def x(self):
        return self.xs[0].x

    @property
    def y(self):
        if self.ndim<2:
            raise ValueError("Dimension y not defined for ndim<2.")
        return self.xs[1].x

    def forward(self):
        '''
        Full forward transform to homogeneous field v
        '''
        vhat = self.v
        for axis in range(self.ndim):
            vhat = self.forward_fft(vhat,axis=axis)
        self.vhat = vhat

    def backward(self):
        '''
        Full backward transform to homogeneous field v
        '''
        v = self.vhat
        for axis in range(self.ndim):
            v = self.backward_fft(v,axis=axis)
        self.v = v

    # ---------------------------------------------------------
    #     Split field in homogeneous and inhomogeneous part
    # ---------------------------------------------------------

    def add_field_bc(self,field_bc):
        assert isinstance(field_bc,FieldBC)
        self.field_bc = field_bc

    def make_homogeneous(self):
        '''
        Subtract inhomogeneous field  (in self.field_bc) from v 
        '''
        import warnings
        if self.field_bc is None:
            warnings.warn("""No inhomogeneous field found. 
                Call add_bc(bchat,axis) first!""")
        else:
            assert self.v.shape == self.inhomogeneous.shape, \
            "Shape mismatch in make_homogeneous"
        return self.v - self.inhomogeneous

    #-----------------------------------------
    #  total = homogeneous + inhomogeneous
    #-----------------------------------------

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

    #-----------------------------------------
    #           Save and animate
    #-----------------------------------------

    def save(self,transform = True):
        if transform: self.backward()
        self.V.append(self.v)
        self.T.append(self.t)

    def dstack(self):
        ''' List to ndarray with order [time,space] '''
        self.VS = np.rollaxis( np.dstack(self.V).squeeze(), -1)
        self.TS = np.rollaxis( np.dstack(self.T).squeeze(), -1)

    def animate(self,x=None,y=None,skip=1,**kwargs):
        self.dstack()

        if x is None:
            if hasattr(self,"x"):
                x = self.x
            else:
                raise ValueError("Can't animate. x not known.") 

        if self.ndim==1:
            return animate_line(x,self.VS[::skip],**kwargs)

        if self.ndim==2:
            if y is None:
                if hasattr(self,"y"):
                    y = self.y
                else:
                    raise ValueError("Can't animate. y not known.") 

            return animate_wireframe(x,y,self.VS[::skip],**kwargs)


class FieldBC(SpectralSpaceBC):
    '''
    Handle inhomogeneous field from inhomogeneous boundary
    conditions. 
    
    Example
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    from pypde.field import *
    N,M = 40,30
    shape = (N,M)

    # Field BC
    field_bc = FieldBC(shape,("CD","CN"),axis=0)
    xx,yy = np.meshgrid(field_bc.x,field_bc.y,indexing="ij")

    bc = np.zeros((2,M))
    bc[0,:] = -1+np.sin(4*field_bc.y)
    bc[1,:] =  1+np.sin(4*field_bc.y)
    field_bc.add_bc(bc)

    from pypde.plot.wireframe import plot
    plot(xx,yy,field_bc.v)
    
    # Derivative along x
    field_bc_deriv = derivative_field(field_bc,deriv=(1,0))
    plot(xx,yy,field_bc_deriv.v)

    # Derivative along y
    field_bc_deriv = derivative_field(field_bc,deriv=(0,1))
    plot(xx,yy,field_bc_deriv.v)
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    def __init__(self,shape,bases,axis):
        SpectralSpaceBC.__init__(self,shape,bases,axis)
        # Field variable
        self.v = np.zeros(self.shape_physical)
        self.vhat = np.zeros(self.shape_spectral)

    def forward(self):
        '''
        Full forward transform to homogeneous field v
        '''
        vhat = self.v
        for axis in range(self.ndim):
            vhat = self.forward_fft(vhat,axis=axis)
        self.vhat = vhat

    def backward(self):
        '''
        Full backward transform to homogeneous field v
        '''
        v = self.vhat
        for axis in range(self.ndim):
            v = self.backward_fft(v,axis=axis)
        self.v = v

    @property
    def x(self):
        return self.xs[0].x

    @property
    def y(self):
        if self.ndim<2:
            raise ValueError("Dimension y not defined for ndim<2.")
        return self.xs[1].x

    def add_bc(self,bc):
        '''
        Input bc must be forward transformed only along self.axis.
        The other dimensions are still in physical space
        '''
        expected_shape = list(self.shape_physical)
        expected_shape[self.axis] = self.shape_spectral[self.axis]
        assert bc.shape == tuple(expected_shape)
        # Transform to real space
        bc = self.backward_fft(bc,axis=self.axis)
        self.v = bc 
        self.forward()



    # @property
    # def v(self):
    #     return self._v
     
    # @v.setter
    # def v(self,value):
    #     self._vhat = self.forward_fft(value,self.axis)
    #     self._v = value
    
    # @property
    # def vhat(self):
    #     return self._vhat

    # @vhat.setter
    # def vhat(self,value):
    #     self._v = self.backward_fft(value,self.axis)
    #     self._vhat = value


    
def derivative_field(field,deriv,out_array = None):
    '''
    Find derivative of field

    Example: 

    # Set field
    shape = (30,20)
    field = Field(shape,("CD","CN"))
    xx,yy = np.meshgrid(field.x,field.y,indexing="ij")

    f = np.sin(np.pi* xx)*np.sin(np.pi*yy)
    field.v = f
    field.forward()

    # Get derivative
    deriv_field = derivative_field(field,deriv=(1,0))
    deriv_field.backward()

    from pypde.plot.wireframe import plot
    plot(xx,yy,field.v)
    plot(xx,yy,deriv_field.v)
    '''
    assert isinstance(field,(Field,FieldBC))
    if isinstance(deriv,int): deriv = (deriv,) # to tuple
    assert field.ndim == len(deriv)
    
    dvhat = field.vhat
    for axis in range(field.ndim):
        dvhat = field.derivative(dvhat,deriv[axis],axis=axis)
    
    if out_array is None:
        bases = [field.xs[0].family_id for i in range(field.ndim)]
        field_deriv = Field(field.shape_physical,tuple(bases))
        field_deriv.vhat = dvhat
        field_deriv.backward()
        return field_deriv
    else:
        out_array[:] = dvhat