import numpy as np
from numpy import pi
from .spectralbase import MetaBase
from .dmsuite import fourdif
from .memoize import memoized

class Fourier(MetaBase):
    """
    Function space for Fourier transform (R2C)
    .. math::
        s_k = a/2 + ak*(cos(k*x)) + bk*(sin(k*x))
        x_k = 2*np.pi*k/N; k=0..N
    
    Parameters:
        N: int
        Number of grid points
        
    """
    def __init__(self,N):
        assert N%2==0, "Fourier dim should be even"
        x = 2*pi*np.arange(N)/N
        MetaBase.__init__(self,N,x)
        self.id = "FO" 
        self.family_id = "FO"

    @property
    def _k(self):
        return np.fft.rfftfreq(self.N,d=(1.0 / self.N )).astype(int)

    @property
    def k(self):
        ''' wavenumber vector '''
        k = self._k
        if self.N % 2 == 0:
            k[self.N//2] = 0 
        return k

    def forward_fft(self,f,axis=0):
        return np.fft.rfft(f,axis=axis)

    def backward_fft(self,c,axis=0):
        return np.real(  np.fft.irfft(c,axis=axis) )

    def get_basis(self, i=0, x=None):
        if x is None: x = self.x
        k = self._k[i]
        return np.exp(1j*x*k)

    def get_basis_derivative(self, i=0, k=0, x=None):
        l = self._k[i]
        output = ((1j*l)**k)*self.get_basis(i,x)
        return output

    def slice(self):
        ''' R2C '''
        return slice(0, self.N//2+1)

    @memoized
    def dms(self,deriv):
        ''' 
        Fourier differentation matrix, applied in spectral space.
        '''
        return np.diag((1j*self._k)**deriv)

    @memoized
    def dmp_collocation(self,deriv):
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
            return self.dmp_collocation(deriv)@f

    # @staticmethod
    # def _discard(A):
    #     ''' Discard'''
    #     return A[:,:]

    # @memoized
    # def pseudoinverse(self,deriv,discardrow=None):
    #     ''' 
    #     Pseudoinverse of dmat_spectral dms. Since dms is diagonal
    #     the inverse will be B = 1/diag(D) but with a zero element on B[0,0]
    #     '''
    #     if discardrow is None: discardrow = 1
    #     k_inv = [0 if i==0 else 1/(1j*k)**deriv for i,k in enumerate(self._k)]
    #     rv = np.diag( k_inv )
    #     return rv[discardrow:,1:]