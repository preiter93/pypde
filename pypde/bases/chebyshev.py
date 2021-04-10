import numpy as np
from scipy.fftpack import dctn
from scipy.sparse import diags
from ..utils.memoize import memoized
from .inner import inner
from .dmsuite import chebdif

def gauss_lobatto(n):
    ''' Return Chebyshev-Gauss-Lobatto grid points'''
    k = np.linspace(n, 0, n + 1)
    return np.sin(np.pi*(n - 2 * k)/(2 * n))

@memoized
def diffmat_spectral(N,deriv):
    '''Derivative matrix in spectral space of classical Chebyshev 
    polynomial on Gauss Lobattor points, see
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
    of deriv'th derivative of classical Chebyshev polynomial
    on Gauss Lobattor points

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

class Spectralbase():
    ''' 
    TODO: Transfer to Metaclass named Metabase
    '''
    def __init__(self,N):
        self.N = N
        self._x = gauss_lobatto(N-1)
    
    @property
    def x(self):
        return self._x

    def _mass(self):
        ''' 
        Mass <TiTj> of Cheby Gauss Lobatto Quad, equvalent to inner(self,self)
        '''
        raise NotImplementedError

    def _mass_inv(self):
        ''' Inverse of _mass'''
        raise NotImplementedError

    def project(self,f):
        ''' Transform to spectral space:
        cn = <Ti,Tj>^-1 @ <Tj,f> where <Ti,Tj> is (sparse) mass matrix'''
        c,sl = np.zeros(self.N), self.slice()
        c[sl] = self._mass_inv()@inner(self,f)
        return c

    def eval(self,c):
        ''' Evaluate f(x) from spectral coefficients c '''
        y = np.zeros(self.N) 
        for i in range(self.N):
            y += c[i]*self.get_basis(i)
        return y

    def slice(self):
        return slice(0, self.N)

    def iter_basis(self,sl=None):
        ''' Return iterator over all bases '''
        if sl is None: sl=self.slice()
        return (self.get_basis(i) 
            for i in range(self.N)[self.slice()])

    def _evaluate_mass_bruteforce(self):
        ''' Return mass matrix <Ti,Tj> by brute inner product.
        Note, <Ti,Tj> is usually sparse, better implement for each child'''
        return inner(self,self)

    def _mass_inv_bruteforce(self):
        return np.linalg.inv( self._evaluate_mass_bruteforce() )


class Chebyshev(Spectralbase):
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
        Spectralbase.__init__(self,N)
    
    def get_basis(self, i=0, x=None):
        if x is None: x = self.x
        w = np.arccos(x)
        return np.cos(i*w)
    

    def derivative(self,f,deriv,method="fft"):
        ''' Calculate derivative of input array f'''
        if method in ("fft", "spectral"):
            c = self.forward_fft(f)
            dc = diff_recursion_spectral(c,deriv)
            #dc = diffmat_spectral(self.N,deriv)@c
            return self.backward_fft(dc)
        elif method in ("dm", "physical"):
            return self.get_deriv_mat(deriv)@f
        else: 
            raise NotImplementedError("Not implemented method: {:s}".format(method))

    @memoized
    def get_deriv_mat(self,deriv):
        return chebdif(self.N,deriv)[1]
    
    def _mass(self):
        return diags([1.0, *[0.5]*(self.N-2), 1.0],0)
    
    def _mass_inv(self):
        return diags([1.0, *[2.0]*(self.N-2), 1.0],0)

    def forward_fft(self,f,nomass=False):
        '''  Transform to spectral space via DCT, similar to project(), see
        https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform ''' 
        c = 0.5*dctn(f,type=1)/(self.N-1)
        c *= [(-1)**k for k in np.arange(self.N)]
        if nomass:
            return c
        else:
            return self._mass_inv()@c
    
    def backward_fft(self,c):
        '''  Transform to physical space via DCT ''' 
        f = c*[(-1)**k for k in np.arange(self.N)]
        f[[0,-1]] *= 2 # compensate factor of dctn type 1 prefactors
        return 0.5*dctn(f,type=1)

class ChebDirichlet(Spectralbase):
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
        Spectralbase.__init__(self,N)
        
        #self.bc = BoundaryValues(self, bc=bc) # TODO

    def eval(self,c):
        ''' Evaluate f(x) from spectral coefficients c '''
        y = np.zeros(self.N) 
        for i in range(self.N):
            y += c[i]*self.get_basis(i)
        return y
            
    def get_basis(self, i=0, x=None):
        if x is None: x = self.x
        if i < self.N-2:
            w = np.arccos(x)
            return np.cos(i*w) - np.cos((i+2)*w)
        elif i == self.N-2:
            return 0.5*(1-x)
        elif i == self.N-1:
            return 0.5*(1+x)
        
    def _mass(self):
        #raise NotImplementedError
        return self._evaluate_mass_bruteforce()

    def _mass_inv(self):
        #raise NotImplementedError
        return self._mass_inv_bruteforce()

    def slice(self):
        ''' Chebdirichlet space defined for [0,N-3] bases + 2 BCs.'''
        return slice(0, self.N-2)

    def forward_fft(self,f):
        '''  Transform to spectral space via DCT '''
        c = Chebyshev.forward_fft(self,f,nomass=True)
        c[self._s0] -= c[self._s1]
        c[self.slice()] = self._mass_inv()@c[self.slice()]
        c[ [-2,-1] ] = 0 # BCs
        return c
    
    def backward_fft(self,c):
        '''  Transform to physical space via DCT ''' 
        c0,c1 = np.zeros(self.N),np.zeros(self.N)
        c0[self._s0],c1[self._s1] = c[self._s0],c[self._s0]
        f0 = Chebyshev.backward_fft(self,c0)
        f1 = Chebyshev.backward_fft(self,c1)
        return f0-f1
    
    @property
    def _s0(self):
        return slice(0, self.N-2)
    
    @property
    def _s1(self):
        return slice(2, self.N)