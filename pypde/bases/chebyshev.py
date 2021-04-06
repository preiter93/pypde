import numpy as np
from scipy.fftpack import dctn
from ..utils.memoize import memoized
from ..utils.inner import inner
from .dmsuite import chebdif

def gauss_lobatto(n):
    ''' Return Chebyshev-Gauss-Lobatto grid points'''
    k = np.linspace(n, 0, n + 1)
    return np.sin(np.pi*(n - 2 * k)/(2 * n))

@memoized
def diffmat_spectral(N,deriv):
    '''Derivative matrix in spectral space, see
    Jan S. Hesthaven - Appendix B p. 256  

    Input:
        N: int
            Number of grit points
        deriv: int
            Order of derivative

    Output:
        ndarray (N x N)
            Derivative matrix, must be applied in spectral
            space to chebyshev coefficients array
            '''
    D = np.zeros( (N,N) )
    if deriv==1:
        for n in range(N):
            for p in range(n+1,N):
                if (p+n)%2!=0: D[n,p] = p*2
    if deriv==2:
        for n in range(N):
            for p in range(n+2,N):
                if (p+n)%2==0:
                    D[n,p] = p*(p**2-n**2)
    if deriv>2:
        raise NotImplementedError("derivatives larger 2 not implemented")
    D[0,:] *= 0.5
    return D

def diff_recursion_spectral(c,deriv):
    ''' Recursion formula for computing coefficients 
    of deriv'th derivative

    Input:
        c: ndarray (dim 1)
            Chebyshev spectral coefficients
        deriv: int
            Order of derivative

    Output:
        ndarray (dim 1)
            Chebyshev spectral ceofficients of derivative
    '''
    N = c.size
    a = np.zeros((N,deriv+1))
    a[:,0] = c
    for ell in np.arange(1,deriv+1):
        a[N-ell-1,ell]=2*(N-ell)*a[N-ell,ell-1];
        for k in np.arange(N-ell-2,0,-1):
            a[k,ell]=a[k+2,ell]+2*(k+1)*a[k+1,ell-1]
        a[0,ell]=a[1,ell-1]+a[2,ell]/2
    return a[:,deriv]

class ChebyBase():
    def __init__(self,N):
        self.N = N
        self._x = gauss_lobatto(N-1)
    
    @property
    def x(self):
        return self._x

    def mass(self):
        return inner(self,self,w=self.w)
    
    @memoized
    def mass_inv(self):
        return np.linalg.inv(self.mass())

    def project_via_mass(self,f):
        return self.mass_inv()@inner(self,f)

    @property
    def w(self):
        ''' Weight for Chebyshev Gauss Lobatto Quadrature'''
        return  np.concatenate(([0.5],np.ones(self.N-2),[0.5] ))

    def eval(self,c):
        y = np.zeros(self.N) 
        for i in range(self.N):
            y += c[i]*self.get_basis(i)
        return y

class Chebyshev(ChebyBase):
    """
    Function space for Chebyshev polynomials
    .. math::
        \phi_k = T_k = cos(k*arccos(x))
        x_k = co(pi*k/N); k=0..N
    
    Parameters:
        N: int
        Number of grid points
        
    Literature: 
    https://www.math.purdue.edu/~shen7/pub/LegendreG.pdf
    https://github.com/spectralDNS/shenfun
    """
    def __init__(self,N):
        ChebyBase.__init__(self,N)
    
    def get_basis(self, i=0, x=None):
        if x is None: x = self.x
        w = np.arccos(x)
        return np.cos(i*w)
    
    def project(self,f):
        ''' Transform to spectral space cn=<Tn(x)*f(x)>_w'''
        return inner(self,f,self.w)*2*self.w
    
    def forward_fft(self,f):
        '''  Transform to spectral space via DCT, see
        https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform ''' 
        c = dctn(f,type=1)*self.w/(self.N-1)
        return c*[(-1)**k for k in np.arange(self.N)]
    
    def backward_fft(self,c):
        '''  Transform to physical space via DCT ''' 
        c = c*[(-1)**k for k in np.arange(self.N)]/self.w
        return 0.5*dctn(c,type=1)

    def derivative(self,f,deriv,method="fft"):
        if method in ("fft", "spectral"):
            fhat = self.forward_fft(f)
            dfhat = diff_recursion_spectral(fhat,deriv)
            return self.backward_fft(dfhat)
        elif method in ("dm", "physical"):
            return self.get_deriv_mat(deriv)@f
        else: 
            raise NotImplementedError("Not implemented method: {:s}".format(method))

    @memoized
    def get_deriv_mat(self,deriv):
        return chebdif(self.N,deriv)[1]


class ChebDirichlet(ChebyBase):
    """
    Function space for Dirichlet boundary conditions
    .. math::
        \phi_k = T_k - T_{k+2}
    
    Parameters:
        N: int
            Number of grid points
        bc: 2-tuple of floats, optional
            Boundary conditions at, respectively, x=(-1, 1).
            
    """
    def __init__(self,N,bc=(0,0)):
        Chebyshev.__init__(self,N)
        
        #self.bc = BoundaryValues(self, bc=bc) # TODO
        
    def get_basis(self, i=0, x=None):
        if x is None: x = self.x
        if i < self.N-2:
            w = np.arccos(x)
            return np.cos(i*w) - np.cos((i+2)*w)
        elif i == self.N-2:
            return 0.5*(1-x)
        elif i == self.N-1:
            return 0.5*(1+x)

    def project(self,f):
        raise NotImplementedError