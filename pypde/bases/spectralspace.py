from .spectralbase import *


class SpectralSpace:
    """
    Class that handles all spectral bases and generalizes
    them to multidimensional spaces

    Input
        bases: list with class Metabase
    """

    def __init__(self, bases):
        if isinstance(bases, MetaBase):
            bases = [bases]
        assert np.all(isinstance(i, MetaBase) for i in bases)
        self._set_bases(bases)
        self.ndim = len(self.shape_physical)
        self.shape = self.shape_physical
        # self.create_dealiased_space(bases)

    def _set_bases(self, bases):
        self.xs = []
        shape_physical = []
        shape_spectral = []
        for i, b in enumerate(bases):
            self.xs.append(b)
            shape_physical.append(self.xs[i].N)
            shape_spectral.append(self.xs[i].M)
        self.shape_physical = tuple(shape_physical)
        self.shape_spectral = tuple(shape_spectral)

    def forward_fft(self, v, axis):
        assert isinstance(axis, int)
        # for axis,x in enumerate(self.xs):
        if axis == 0:
            vhat = self.xs[axis].forward_fft(v)
        else:
            vhat = np.swapaxes(v, axis, 0)
            vhat = self.xs[axis].forward_fft(vhat)
            vhat = np.swapaxes(vhat, axis, 0)
        return vhat

    def backward_fft(self, vhat, axis):
        assert isinstance(axis, int)
        if axis == 0:
            v = self.xs[axis].backward_fft(vhat)
        else:
            v = np.swapaxes(vhat, axis, 0)
            v = self.xs[axis].backward_fft(v)
            v = np.swapaxes(v, axis, 0)
        return v

    def derivative(self, vhat, deriv, axis, out_cheby=True):
        if axis == 0:
            return self.xs[axis].derivative(vhat, deriv, out_cheby)
        else:
            vhat = np.swapaxes(vhat, axis, 0)
            dvhat = self.xs[axis].derivative(vhat, deriv, out_cheby)
            return np.swapaxes(dvhat, axis, 0)

    # def create_dealiased_space(self,bases):
    #     if np.all([hasattr(i,"dealias") for i in bases]):
    #         print("yes")
    #         dealiased_bases = [i.dealias for i in bases]
    #         self.dealias = SpectralSpace(dealiased_bases)


class SpectralSpaceBC(SpectralSpace):
    """
    Handles a boundary conditions along 1 axis
    This class is used internally in field.py

    Input
        bases: list with class Metabase

        axis: int
            Axis along which bc is applied
    """

    def __init__(self, bases, axis):
        SpectralSpace.__init__(self, bases)
        self.axis = axis
        self._check_axis_bases()

    def _check_axis_bases(self):
        """
        Bases along self.axis should be should be DirichletC or NeumannC
        and not implement self.bc
        """
        if hasattr(self.xs[self.axis], "bc"):
            self.xs[self.axis] = self.xs[self.axis].bc
            SpectralSpace.__init__(self, self.xs)
