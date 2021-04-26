import numpy as np
from .bases import *

class SpectralSpace():
    '''
    Class that handles all spectral bases and generalizes
    them to multidimensional spaces
    
    Input
        shape: int tuple (ndim)
            Shape of field in real space, can be 1d or 2d
        bases: str tuple
            Define the spectral bases.
            Chebyshev:
                "CH" or "Chebyshev"
                "CD" or "ChebDirichlet"
                "CN" or "ChebNeumann"
            Fourier:
                "FO" or "Fourier"
    '''
    def __init__(self,shape,bases):
        shape,bases = self._check_input(shape,bases)
        assert len(shape) == len(bases), "Shape size must match number of bases"
        self.shape = shape
        self.dim = len(self.shape)
        self._set_bases(bases)
        
    def _check_input(self,shape,bases):
        if isinstance(shape, int): shape = (shape,)
        if isinstance(bases, str): bases = (bases,)
        return shape, bases
    
    def _set_bases(self,bases):
        self.xs = []
        shape_spectral = []
        for i,key in enumerate(bases):
            N = self.shape[i]
            self.xs.append( self._bases_from_key(key)(N) )
            shape_spectral.append(self.xs[i].M)
        self.shape_spectral = tuple(shape_spectral)
            
    def _bases_from_key(self,key):
        if key == "CH" or key == "Chebyshev":
            return Chebyshev
        elif key == "CD" or key == "ChebDirichlet":
            return ChebDirichlet
        elif key == "CN" or key == "ChebNeumann":
            return ChebNeumann
        elif key == "FO" or key == "Fourier":
            return Fourier
        else:
            raise ValueError("Key {:} not available.".format(key))

    def forward_fft(self,v):
        assert v.shape == self.shape
        vhat = v
        for axis,x in enumerate(self.xs):
            if axis == 0:
                vhat = x.forward_fft(vhat)
            else:
                vhat = np.swapaxes(vhat,axis,0)
                vhat = x.forward_fft(vhat)
                vhat = np.swapaxes(vhat,axis,0)
        return vhat
    
    def backward_fft(self,vhat):
        ''' TODO: Add BCs '''
        assert vhat.shape == self.shape_spectral
        v = vhat
        for axis,x in enumerate(self.xs):
            if axis == 0:
                v = x.backward_fft(v)
            else:
                v = np.swapaxes(v,axis,0)
                v = x.backward_fft(v)
                v = np.swapaxes(v,axis,0)
        return v