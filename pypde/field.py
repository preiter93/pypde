import numpy as np
from .bases.spectralspace import *
from .bases.memoize import memoized
from .bases.utils import zero_pad, zero_unpad
from .plot.anim import animate_line, animate_contour, animate_wireframe
import h5py


class FieldBase:
    """
    Functions that are shared by Field and FieldBC
    """

    def forward(self, v=None, undealias_after=None):
        """
        Full forward transform to homogeneous field v
        """
        if v is None:
            vhat = self.v
        else:
            vhat = v

        if undealias_after is None:
            undealias_after = self.dealiased_space

        for axis in range(self.ndim):
            vhat = self.forward_fft(vhat, axis=axis)
            if undealias_after:
                vhat = zero_unpad(vhat, self.size_undealiased[axis], axis=axis)

        if v is None:
            self.vhat = vhat
        else:
            return vhat

    def backward(self, vhat=None, dealias_before=None):
        """
        Full backward transform to homogeneous field v
        """
        if vhat is None:
            v = self.vhat
        else:
            v = vhat

        if dealias_before is None:
            dealias_before = self.dealiased_space

        for axis in range(self.ndim):
            if dealias_before:
                v = zero_pad(v, self.xs[axis].M, axis=axis)
            v = self.backward_fft(v, axis=axis)

        if vhat is None:
            self.v = v
        else:
            return v

    @property
    def x(self):
        return self.xs[0].x

    @property
    def y(self):
        if self.ndim < 2:
            raise ValueError("Dimension y not defined for ndim<2.")
        return self.xs[1].x

    @property
    def dx(self):
        xm = np.zeros(self.x.size + 1)
        xm[0], xm[-1] = self.x[0], self.x[-1]
        xm[1:-1] = (self.x[1:] + self.x[:-1]) / 2.0
        return np.diff(xm)

    @property
    def dy(self):
        ym = np.zeros(self.y.size + 1)
        ym[0], ym[-1] = self.y[0], self.y[-1]
        ym[1:-1] = (self.y[1:] + self.y[:-1]) / 2.0
        return np.diff(ym)

    # -- Read Write
    def write(
        self,
        filename="file_0.h5",
        dict=None,
        leading_str="flow",
        add_time=True,
        grp_name="",
    ):
        # -- Filename
        if filename is None:
            filename = leading_str
            if add_time:
                filename = filename + "_{:07.2f}".format(self.t)
            filename = filename + ".h5"

        if grp_name and grp_name[-1] != "/":
            grp_name = grp_name + "/"
        v = grp_name + "v"
        vhat = grp_name + "vhat"

        # -- Write
        print("Write {:s} ...".format(filename))
        hf = h5py.File(filename, "a")
        self.write_single_hdf5(hf,v,self.v)
        self.write_single_hdf5(hf,vhat,self.vhat)
        self.write_single_hdf5(hf,"time",self.t)

        if self.x is not None:
            self.write_single_hdf5(hf,"x",self.x)
        if self.y is not None:
            self.write_single_hdf5(hf,"y",self.y)
        if dict is not None:
            for key in dict:
                self.write_single_hdf5(hf,key,dict[key])
        # -- Close
        hf.close()

    @staticmethod
    def write_single_hdf5(hf,name,data):
        if not name in hf:
            hf.create_dataset(name, data=data)
        else:
            #print("Data {:} exists already. Overwrite...".format(name))
            data = hf[name]       # load the data
            data[...] =data       # assign new values to data

    @staticmethod
    def read_single_hdf5(hf,name):
        data = hf.get(name)
        if data is not None:
            return np.array(data)
        else:
            print("Cannot read {:} ...".format(name))
            print("Following keys exist: ")
            hf.visit(print_grp)
            return 0.

    def read(
        self,
        filename="file_0.h5",
        dict=None,
        leading_str="flow",
        add_time=True,
        grp_name="",
    ):
        # -- Filename
        if filename is None:
            filename = leading_str
            if add_time:
                filename = filename + "_{:07.2f}".format(self.t)
            filename = filename + ".h5"

        # -- Read
        print("Read {:s} ...".format(filename))
        hf = h5py.File(filename, "r")

        if grp_name and grp_name[-1] != "/":
            grp_name = grp_name + "/"
        v = grp_name + "v"
        vhat = grp_name + "vhat"

        # -- Read
        self.v[:] = self.read_single_hdf5(hf,v)
        self.vhat[:] = self.read_single_hdf5(hf,vhat)
        self.t = self.read_single_hdf5(hf,"time")

        if dict is not None:
            for key in dict:
                dict[key] = self.read_single_hdf5(hf,key)
        # -- Close
        hf.close()


def print_grp(name):
    print("Group:", name)


class Field(SpectralSpace, FieldBase):
    """
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
    """

    def __init__(self, bases):
        if isinstance(bases, MetaBase):
            bases = [bases]
        SpectralSpace.__init__(self, bases)
        FieldBase.__init__(self)
        self.bases = bases

        self.v = np.zeros(self.shape_physical)  # physical field
        self.vhat = np.zeros(self.shape_spectral)  # spectral field

        self.field_bc = None  # inhomogeneous field

        self.t = 0  # time
        self.V = []  # Storage Field
        self.T = []  # Storage Time

        # create deliased field
        self.dealiased_space = False
        self.create_dealiased_field(bases)

    def create_dealiased_field(self, bases):
        if np.all([hasattr(i, "dealias") for i in bases]):
            dealiased_field = [i.dealias for i in bases]
            self.dealias = Field(dealiased_field)
            self.dealias.size_undealiased = [self.xs[i].M for i in range(self.ndim)]
            self.dealias.dealiased_space = True

    # ---------------------------------------------------------
    #     Split field in homogeneous and inhomogeneous part
    # ---------------------------------------------------------

    def add_field_bc(self, field_bc):
        assert isinstance(field_bc, FieldBC)
        self.field_bc = field_bc

    def make_homogeneous(self):
        """
        Subtract inhomogeneous field  (in self.field_bc) from v
        """
        import warnings

        if self.field_bc is None:
            warnings.warn(
                """No inhomogeneous field found.
                Call add_bc(bchat,axis) first!"""
            )
        else:
            assert (
                self.v.shape == self.inhomogeneous.shape
            ), "Shape mismatch in make_homogeneous"
        return self.v - self.inhomogeneous

    # -----------------------------------------
    #  total = homogeneous + inhomogeneous
    # -----------------------------------------

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

    # -----------------------------------------
    #           Save and animate
    # -----------------------------------------

    def save(self, transform=True):
        if transform:
            self.backward()
        self.V.append(self.v)
        self.T.append(self.t)

    def dstack(self):
        """List to ndarray with order [time,space]"""
        self.VS = np.rollaxis(np.dstack(self.V).squeeze(), -1)
        self.TS = np.rollaxis(np.dstack(self.T).squeeze(), -1)

    def animate(self, x=None, y=None, skip=1, wireframe=False, **kwargs):
        self.dstack()

        if x is None:
            if hasattr(self, "x"):
                x = self.x
            else:
                raise ValueError("Can't animate. x not known.")

        if self.ndim == 1:
            return animate_line(x, self.VS[::skip], **kwargs)

        if self.ndim == 2:
            if y is None:
                if hasattr(self, "y"):
                    y = self.y
                else:
                    raise ValueError("Can't animate. y not known.")
            if wireframe:
                return animate_wireframe(x, y, self.VS[::skip], **kwargs)
            return animate_contour(x, y, self.VS[::skip], **kwargs)


class FieldBC(SpectralSpaceBC, FieldBase):
    """
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
    """

    def __init__(self, bases, axis):
        SpectralSpaceBC.__init__(self, bases, axis)
        FieldBase.__init__(self)
        # Field variable
        self.v = np.zeros(self.shape_physical)
        self.vhat = np.zeros(self.shape_spectral)

        self.dealiased_space = False

    def add_bc(self, bc):
        """
        Input bc must be forward transformed only along self.axis.
        The other dimensions are still in physical space
        """
        expected_shape = list(self.shape_physical)
        expected_shape[self.axis] = self.shape_spectral[self.axis]
        assert bc.shape == tuple(expected_shape)
        # Transform to real space
        bc = self.backward_fft(bc, axis=self.axis)
        self.v = bc
        self.forward()


class MultiField:
    """
    Simple Class that collects multiple
    fields and defines collective routines
    """

    def __init__(self, fields, names):
        self.fields = []
        self.names = []
        for f, n in zip(fields, names):
            if not isinstance(f, Field):
                raise ValueError("Must be of type Field.")
            self.fields.append(f)
            self.names.append(n)

    def save(self):
        for f in self.fields:
            f.save()

    def update_time(self, dt):
        for f in self.fields:
            f.t += dt

    def read(self, filename=None, leading_str="", add_time=True, dict={}):
        for f, n in zip(self.fields, self.names):
            f.read(
                filename=filename,
                leading_str=leading_str,
                add_time=add_time,
                dict=dict,
                grp_name=n,
            )

    def write(self, filename=None, leading_str="", add_time=True, dict={}):
        for f, n in zip(self.fields, self.names):
            f.backward()
            f.write(
                filename=filename,
                leading_str=leading_str,
                add_time=add_time,
                dict=dict,
                grp_name=n,
            )

    def interpolate(self, old_fields, spectral=True):
        from pypde.field_operations import interpolate

        for f, f_old in zip(self.fields, old_fields.fields):
            interpolate(f_old, f, spectral)
