import numpy as np
from .bases.spectralspace import *
from .bases.memoize import memoized
from .bases.utils import zero_pad,zero_unpad
from .plot.anim import animate_line,animate_contour,animate_wireframe


class FieldBase():
    '''
    Functions that are shared by Field and FieldBC
    '''

    def forward(self,v=None, undealias_after=None):
        '''
        Full forward transform to homogeneous field v
        '''
        if v is None:
            vhat = self.v
        else:
            vhat = v

        if undealias_after is None:
            undealias_after = self.dealiased_space

        for axis in range(self.ndim):
            vhat = self.forward_fft(vhat,axis=axis)
            if undealias_after:
                vhat = zero_unpad(vhat,self.size_undealiased[axis],axis=axis)


        if v is None:
            self.vhat = vhat
        else:
            return vhat

    def backward(self,vhat=None, dealias_before=None):
        '''
        Full backward transform to homogeneous field v
        '''
        if vhat is None:
            v = self.vhat
        else:
            v = vhat

        if dealias_before is None:
            dealias_before = self.dealiased_space

        for axis in range(self.ndim):
            if dealias_before:
                v = zero_pad(v,self.xs[axis].M,axis=axis)
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
        if isinstance(bases, MetaBase): bases = [bases]
        SpectralSpace.__init__(self,bases)
        FieldBase.__init__(self)
        self.bases = bases

        self.v = np.zeros(self.shape_physical)      # physical field
        self.vhat = np.zeros(self.shape_spectral)   # spectral field

        self.field_bc = None #inhomogeneous field

        self.t = 0      # time
        self.V = []     # Storage Field
        self.T = []     # Storage Time

        # create deliased field
        self.dealiased_space = False
        self.create_dealiased_field(bases)

    def create_dealiased_field(self,bases):
        if np.all([hasattr(i,"dealias") for i in bases]):
            dealiased_field = [i.dealias for i in bases]
            self.dealias = Field(dealiased_field)
            self.dealias.size_undealiased = [self.xs[i].M for i in
            range(self.ndim)]
            self.dealias.dealiased_space = True
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

        self.dealiased_space = False

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

#-------------------------------------------------------
#          Some useful operations
#-------------------------------------------------------
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
    deriv_field = grad(field,deriv=(1,0),return_field=True)
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
        field_deriv = Field( [field.xs[i].family
            for i in range(field.ndim)] )
        field_deriv.vhat = dvhat
        #field_deriv.backward()
        return field_deriv
    else:
        return dvhat


def cheby_to_galerkin(uhat,galerkin_field):
    for axis in range(uhat.ndim):
        if axis==0:
            uhat = galerkin_field.xs[axis].from_chebyshev(uhat)
        else:
            uhat = np.swapaxes(uhat,0,axis)
            uhat = galerkin_field.xs[axis].from_chebyshev(uhat)
            uhat = np.swapaxes(uhat,0,axis)
    return uhat

def galerkin_to_cheby(vhat,galerkin_field):
    for axis in range(vhat.ndim):
        if axis==0:
            vhat = galerkin_field.xs[axis].to_chebyshev(vhat)
        else:
            vhat = np.swapaxes(vhat,0,axis)
            vhat = galerkin_field.xs[axis].to_chebyshev(vhat)
            vhat = np.swapaxes(vhat,0,axis)
    return vhat

def convective_term(v_field, ux, uz,
deriv_field=None, add_bc=None, dealias=False):
    '''
    Calculate
        ux*dvdx + uz*dvdz

    Input
        v_field: class Field
            Contains field variable vhat in spectral space
        ux,uz:  ndarray
            (Dealiased) velocity fields in physical space
        deriv_field: field (optional)
            Field (space) where derivatives life
        add_bc: ndarray (optional)
            Additional term (physical space), which is added
            before forward transform.
        dealias: bool (optional)
            Dealias convective term. In this case, input ux and
            uz must already be dealiased and deriv_field must
            be initialized with ,dealias=3/2

    Return
        Field of (dealiased) convective term in physical space
        Transform to spectral space via conv_field.forward()
    '''
    if deriv_field is None:
        if dealias:
            deriv_field = Field( [
            Base(v_field.shape[0],"CH",dealias=3/2),
            Base(v_field.shape[1],"CH",dealias=3/2)] )
        else:
            deriv_field = Field( [
            Base(v_field.shape[0],"CH"),
            Base(v_field.shape[1],"CH")] )

    # -- Calculate derivatives of ux and uz

    # dudx
    vhat = grad(v_field,(1,0),return_field=False)
    if dealias:
        dudx = deriv_field.dealias.backward(vhat)
    else:
        dudx = deriv_field.backward(vhat)

    # dvdz
    vhat = grad(v_field,(0,1),return_field=False)
    if dealias:
        dvdz = deriv_field.dealias.backward(vhat)
    else:
        dvdz = deriv_field.backward(vhat)

    conv = dudx*ux + dvdz*uz

    if add_bc is not None:
        conv += add_bc

    if dealias:
        return deriv_field.dealias.forward(conv)
    return deriv_field.forward(conv)
# def grad(field,deriv, return_field=False):
#     return derivative_field(field,deriv, return_field=return_field)
