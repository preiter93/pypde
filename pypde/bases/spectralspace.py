from .spectralbase import *

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

    def forward_fft(self,v,axis):
        assert isinstance(axis,int)
        #for axis,x in enumerate(self.xs):
        if axis == 0:
            vhat = self.xs[axis].forward_fft(v)
        else:
            vhat = np.swapaxes(v,axis,0)
            vhat = self.xs[axis].forward_fft(vhat)
            vhat = np.swapaxes(vhat,axis,0)
        return vhat

    def backward_fft(self,vhat,axis):
        assert isinstance(axis,int)
        if axis == 0:
            v = self.xs[axis].backward_fft(vhat)
        else:
            v = np.swapaxes(vhat,axis,0)
            v = self.xs[axis].backward_fft(v)
            v = np.swapaxes(v,axis,0)
        return v

class SpectralSpaceBC(SpectralSpace):
    '''
    Handles a boundary conditions along 1 axis
    This class is used internally in field.py

    Input
        shape: tuple
            shape of field in physical space
        bases: tuple
            ChebDirichlet (CD) or Chebneumann (CN)
            support BCs
        axis: int
            Axis along which bc is applied
    '''
    def __init__(self,shape,bases,axis):
        SpectralSpace.__init__(self,shape,bases)
        for b in self.xs:
            assert b.bc is not None
        self.axis = axis

    def forward_fft(self,v,axis):
        '''
        '''
        assert isinstance(axis,int)

        if axis == 0:
            return self.xs[axis].bc.forward_fft(v)
        else:
            vhat = np.swapaxes(v,axis,0)
            vhat = self.xs[axis].bc.forward_fft(vhat)
            vhat = np.swapaxes(vhat,axis,0)
            return vhat

    def backward_fft(self,vhat,axis):
        '''
        '''
        assert isinstance(axis,int)
        assert vhat.shape[self.axis] == 2

        if axis == 0:
            return self.xs[axis].bc.backward_fft(vhat)
        else:
            v = np.swapaxes(vhat,axis,0)
            v = self.xs[axis].bc.backward_fft(v)
            v = np.swapaxes(v,axis,0)
            return v
