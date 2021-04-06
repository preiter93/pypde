'''
Classes for spectral bases
'''
from .dmsuite import *
from .utils.memoize import memoized



class Basis:
    '''
    Base class for spectral bases
    '''
    def __init__(self,N,L):
        self.N = N
        self.L = L

    def get_deriv_mat(self,deriv):
        raise NotImplementedError

    def set_bc(self,D,pos,which="Dirichlet"):
        ''' Replace rows of D to apply BCs '''
        assert D.shape[0] == D.shape[1]
        assert which in ["Dirichlet","Neumann"]

        if which=="Dirichlet":
            B = np.eye( self.N) 
        if which=="Neumann": 
            B = self.get_deriv_mat(1)
        
        if not isinstance(pos, list): 
            pos = [pos]
            
        for p in pos: 
            D[p,:] = B[p,:] # replace

class Chebyshev(Basis):
    '''
    Class for Chebyshev polynomials defined on Gauss-Lobatto 
    points: x = [-1,1]*L
    '''
    def __init__(self,N,L=2):
        Basis.__init__(self,N,L)

    @property
    def x(self):
        return chebdif(self.N,1,L=self.L)[0]
    
    @memoized
    def get_deriv_mat(self,deriv):
        return chebdif(self.N,deriv,L=self.L)[1]

    def deriv_dm(self,f,deriv,axis=0):
        if axis==0:
            return self.get_deriv_mat(deriv)@f
        if axis==1:
            return f@self.get_deriv_mat(deriv)
            
    def deriv_fft(self,f,deriv):
        return chebdifft(f,deriv,L=self.L)

class Fourier(Basis):
    '''
    Class for Fourier polynomials defined on x = [0,2*pi]*L
    '''
    def __init__(self,N,L=2*np.pi):
        Basis.__init__(self,N,L)

    @property
    def x(self):
        return fourdif(self.N,1,L=self.L)[0]

    @property
    def k(self):
        ''' wavenumber vector '''
        return np.fft.fftfreq(self.N,d=(1.0 / self.N ))
    
    @memoized
    def get_deriv_mat(self,deriv):
        return fourdif(self.N,deriv,L=self.L)[1]

    def deriv_dm(self,f,deriv,axis=0):
        if axis==0:
            return self.get_deriv_mat(deriv)@f
        if axis==1:
            return f@self.get_deriv_mat(deriv)

    def deriv_fft(self,f,deriv):
        return fourdifft(f,deriv,L=self.L)