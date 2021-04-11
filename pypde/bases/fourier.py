import numpy as np
from numpy import pi
from .spectralbase import SpectralBase
from .dmsuite import fourdif
from ..utils.memoize import memoized

class Fourier(SpectralBase):
    """
    Function space for Fourier 
    .. math::
        s_k = a/2 + ak*(cos(k*x)) + bk*(sin(k*x))
        x_k = 2*np.pi*k/N; k=0..N
    
    Parameters:
        N: int
        Number of grid points
        
    """
    def __init__(self,N):
        x = 2*pi*np.arange(N)/N
        SpectralBase.__init__(self,N,x)

    @property
    def k(self):
        ''' wavenumber vector '''
        return np.fft.fftfreq(self.N,d=(1.0 / self.N ))

    @memoized
    def colloc_deriv_mat(self,deriv):
        ''' Collocation derivative matrix, must be applied in physical space.'''
        return fourdif(self.N,deriv,L=2*pi)[1]

    def derivative(self,f,deriv,method="fft"):
        assert method in ["fft","spectral","dm","physical"]
        ''' Calculate derivative of input array f'''
        if method in ("fft", "spectral"):
            k = (self.k*complex(0,1))**deriv
            c = self.forward_fft(f)
            return self.backward_fft(c*k)
        elif method in ("dm", "physical"):
            return self.colloc_deriv_mat(deriv)@f

    def forward_fft(self,f):
        return np.fft.fft(f)

    def backward_fft(self,c):
        return np.real(  np.fft.ifft(c) )

    def project(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def _mass(self):
        raise NotImplementedError
    
    def _stiff(self):
        raise NotImplementedError

    def get_basis(self):
        raise NotImplementedError

    def get_basis_derivative(self):
        raise NotImplementedError

    def iter_basis(self):
        raise NotImplementedError

    def iter_deriv(self):
        raise NotImplementedError
