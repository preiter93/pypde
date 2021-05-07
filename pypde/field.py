import numpy as np
from .bases.spectralspace import *
from .bases.memoize import memoized
from .plot.anim import animate_line,animate_contour,animate_wireframe


class FieldBase():
    '''
    Functions that are shared by Field and FieldBC
    '''

    def forward(self,v=None):
        '''
        Full forward transform to homogeneous field v
        '''
        if v is None:
            vhat = self.v 
        else:
            vhat = v
        
        for axis in range(self.ndim):
            vhat = self.forward_fft(vhat,axis=axis)
        
        if v is None:
            self.vhat = vhat
        else:
            return vhat

    def backward(self,vhat=None):
        '''
        Full backward transform to homogeneous field v
        '''
        if vhat is None:
            v = self.vhat
        else:
            v = vhat

        for axis in range(self.ndim):
            v = self.backward_fft(v,axis=axis)

        if vhat is None:
            self.v = v
        else:
            return v

    @property
    def x(self):
        return self.xs[0].x

    @property
    def y(self):
        if self.ndim<2:
            raise ValueError("Dimension y not defined for ndim<2.")
        return self.xs[1].x

class Field(SpectralSpace,FieldBase):
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
    xbase = Base(N,"CD")
    ybase = Base(M,"CN")
    field = Field( [xbase,ybase])

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
    field_deriv = grad(field,deriv=(1,0),return_field=True)
    plot(xx,yy,field_deriv.v)

    # Derivative along y (almost zero)
    field_deriv = grad(field,deriv=(0,1),return_field=True)
    plot(xx,yy,field_deriv.v)
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    def __init__(self,bases):
        SpectralSpace.__init__(self,bases)
        FieldBase.__init__(self)
        self.bases = bases

        self.v = np.zeros(self.shape_physical)      # physical field
        self.vhat = np.zeros(self.shape_spectral)   # spectral field

        self.field_bc = None #inhomogeneous field

        self.t = 0      # time
        self.V = []     # Storage Field
        self.T = []     # Storage Time

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

    def animate(self,x=None,y=None,skip=1,wireframe=False,**kwargs):
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
            if wireframe:
                return animate_wireframe(x,y,self.VS[::skip],**kwargs)
            return animate_contour(x,y,self.VS[::skip],**kwargs)


class FieldBC(SpectralSpaceBC,FieldBase):
    '''
    Handle inhomogeneous field from inhomogeneous boundary
    conditions. 
    
    Example
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    from pypde.field import *
    N,M = 40,30
    shape = (N,M)

    # Field BC
    xbase = Base(N,"CD")
    ybase = Base(M,"CN")
    field_bc = FieldBC([xbase,ybase],axis=0)
    xx,yy = np.meshgrid(field_bc.x,field_bc.y,indexing="ij")

    bc = np.zeros((2,M))
    bc[0,:] = -1+np.sin(4*field_bc.y)
    bc[1,:] =  1+np.sin(4*field_bc.y)
    field_bc.add_bc(bc)

    from pypde.plot.wireframe import plot
    plot(xx,yy,field_bc.v)
    
    # Derivative along x
    field_bc_deriv = grad(field_bc,deriv=(1,0),return_field=True)
    plot(xx,yy,field_bc_deriv.v)

    # Derivative along y
    field_bc_deriv = grad(field_bc,deriv=(0,1),return_field=True)
    plot(xx,yy,field_bc_deriv.v)
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    def __init__(self,bases,axis):
        SpectralSpaceBC.__init__(self,bases,axis)
        FieldBase.__init__(self)
        # Field variable
        self.v = np.zeros(self.shape_physical)
        self.vhat = np.zeros(self.shape_spectral)

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

    
def grad(field,deriv, return_field=False):
    '''
    Find derivative of field

    Example: 

    # Set field
    N,M = 40,30
    xbase = Base(N,"CD")
    ybase = Base(M,"CN")
    field = Field( [xbase,ybase] )
    xx,yy = np.meshgrid(field.x,field.y,indexing="ij")

    f = np.sin(np.pi* xx)*np.sin(np.pi*yy)
    field.v = f
    field.forward()

    # Get derivative
    deriv_field = grad(field,deriv=(1,0))
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

    if return_field:
        #bases = [field.xs[0].family_id for i in range(field.ndim)]
        field_deriv = Field( [field.xs[i].family for i in range(field.ndim)] )
        field_deriv.vhat = dvhat
        field_deriv.backward()
        return field_deriv
    else:
        return dvhat

# def grad(field,deriv, return_field=False):
#     return derivative_field(field,deriv, return_field=return_field)