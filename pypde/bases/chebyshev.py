import numpy as np
from scipy.fftpack import dctn
from scipy.sparse import diags
from scipy.sparse.linalg import inv as spinv
from ..utils.memoize import memoized
from .dmsuite import (gauss_lobatto,diff_mat_spectral,
    diff_recursion_spectral,chebdif)
from numpy.polynomial import chebyshev as n_cheb
from .spectralbase import SpectralBase


class Chebyshev(SpectralBase):
    """
    Function space for Chebyshev polynomials
    .. math::
        \phi_k = T_k = cos(k*arccos(x))
        x_k = cos(pi*k/N); k=0..N
    
    Parameters:
        N: int
        Number of grid points
        
    Literature: 
    https://www.math.purdue.edu/~shen7/pub/LegendreG.pdf
    https://github.com/spectralDNS/shenfun
    """
    def __init__(self,N):
        x = gauss_lobatto(N-1)
        SpectralBase.__init__(self,N,x)
        self.id = "CH" 
    
    def get_basis(self, i=0, x=None):
        if x is None: x = self.x
        w = np.arccos(x)
        return np.cos(i*w)

    def get_basis_derivative(self, i=0, k=0, x=None):
        if x is None: x = self.x
        x = np.atleast_1d(x)
        basis = np.zeros(self.N)
        basis[i] = 1
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        return basis(x)
    
    # @property
    # def _mass(self):
    #     ''' Sparse Mass matrix <Ti*Tj>_w,
    #     the same as mass in Parent SpectralBase'''
    #     return diags([1.0, *[0.5]*(self.N-2), 1.0],0)
    
    @property
    def _mass_inv(self):
        return diags([1.0, *[2.0]*(self.N-2), 1.0],0)

    def forward_fft(self,f,mass=True):
        '''  Transform to spectral space via DCT, similar to project(), see
        https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform '''
        sign = np.array([(-1)**k for k in np.arange(self.N)]) 
        c = 0.5*dctn(f,type=1,axes=0)/(self.N-1)
        if len(f.shape)==2:
            c  = c*sign[:, None]  
        else:
            c *= sign
        return self._mass_inv@c if mass else c
    
    def backward_fft(self,c):
        '''  Transform to physical space via DCT ''' 
        # compensate factor of dctn type 1
        sign = np.array([(-1)**k for k in np.arange(self.N)]) 
        if len(c.shape)==2:   
            f = c*sign[:, None]; f[[0,-1],:] *= 2
        else:
            f = c*sign; f[[0,-1]] *= 2 
        return 0.5*dctn(f,type=1,axes=0)

    @memoized
    def colloc_deriv_mat(self,deriv):
        ''' Collocation derivative matrix, must be applied in physical space.'''
        return chebdif(self.N,deriv)[1]

    @memoized
    def spec_deriv_mat(self,deriv):
        ''' 
        Chebyshev differentation matrix. Applied in spectral space.
        Action is equivalent to differentiation via recursion by diff_recursion_spectral
        '''
        return diff_mat_spectral(self.N,deriv)

    def derivative(self,f,deriv,method="fft"):
        assert method in ["fft","spectral","dm","physical"]
        ''' Calculate derivative of input array f'''
        if method in ("fft", "spectral"):
            c = self.forward_fft(f)
            dc = diff_recursion_spectral(c,deriv)
            #dc = diff_mat_spectral(self.N,deriv)@c
            return self.backward_fft(dc)
        elif method in ("dm", "physical"):
            return self.colloc_deriv_mat(deriv)@f


class GalerkinChebyshev(SpectralBase):
    """
    Base class for composite Chebyshev spaces
    like ChebDirichlet used in the Galerkin method.
    
    Define linear combination of Chebyhsev polynomials
    in stencil and set size by slice.

    """
    def __init__(self,N):
        x = gauss_lobatto(N-1)
        SpectralBase.__init__(self,N,x)

    def slice(self):
        ''' 
        Galerkin space usually of size [0,N-3] (+ 2 BCs bases)
        Can be overwritten in child classes
        '''
        return slice(0, self.N-2)

    def stencil(self,inv=False):
        ''' 
        Must be implemented on child class! 
        All Transformations are derived from the stencil.

        Stencil Matrix to transform Coefficients (MxN):      
            Chebyshev (N) -> Galerkin (M)
        Inverse of stencil (NxM): 
            Galerkin (M) -> Chebyshev (N)
        
        Stencil:
            N x N Sparse Matrix (diagonal banded)

        Literature:
            K. Julien: doi:10.1016/j.jcp.2008.10.043 
            https://www.sciencedirect.com/science/article/
            pii/S002199910800569X
        '''
        raise NotImplementedError

    def _to_galerkin(self,cheby_c):
        return self.stencil(False)@cheby_c

    def _to_chebyshev(self,galerkin_c):
        return self.stencil(True)@galerkin_c

    def forward_fft(self,f):
        '''  
        Transform to spectral space via DCT 
        '''
        c = Chebyshev.forward_fft(self,f,mass=False)
        c = self._to_galerkin(c)
        return self._mass_inv@c

    def backward_fft(self,c):
        '''  
        Transform to physical space via DCT 
        ''' 
        c = self._to_chebyshev(c)
        return Chebyshev.backward_fft(self,c)

    def derivative(self,f,deriv,method="fft"):
        ''' 
        Calculate derivative of input array f 
        '''
        assert method in ["fft","spectral"]
        c = self.forward_fft(f)
        c = self._to_chebyshev(c)
        dc = diff_recursion_spectral(c,deriv)
        return Chebyshev.backward_fft(self,dc)

    def get_basis(self,i=0,x=None):
        return self.get_basis_derivative(i=i,k=0,x=x)

    def get_basis_derivative(self, i=0, k=0, x=None):
        if x is None: x = self.x
        x = np.atleast_1d(x)
        if i < self.N-2:
            basis = self.stencil().toarray()[i,:]
            basis = n_cheb.Chebyshev(basis)
            if k > 0:
                basis = basis.deriv(k)
            return basis(x)
        else:
            return self.get_basis_bc(i,k,x)


class ChebDirichlet(GalerkinChebyshev):
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
        GalerkinChebyshev.__init__(self,N)
        self.id = "CD" 

    def stencil(self,inv=False):
        '''  
        Matrix representation of:
            phi_k = T_k - T_{k+2}

        See GalerkinChebyshev for details
        '''
        S = np.zeros((self.M,self.N))
        for i in range(self.M):
            S[i,i],S[i,i+2] = 1, -1
        if inv: return S.T 
        return S

    def get_basis_bc(self,i,k=0,x=None):
        ''' 
        Base functions for N-2 and N-1 to enforce non-zero BCs
            N-2: phi(x) = 0.5*(1-x)
            N-1: phi(x) = 0.5*(1+x)
        '''
        assert i == self.N-2 or i == self.N-1
        if i == self.N-2:
            return 0*x-0.5 if k==1 else 0.5*(1-x) if k==0 else 0*x
        elif i == self.N-1:
            return 0*x+0.5 if k==1 else 0.5*(1+x) if k==0 else 0*x

    # def _mass(self):
    #     ''' 
    #     Eq. (2.5) of Shen - Effcient Spectral-Galerkin Method II.
    #     '''
    #     diag0 = [1.5, *[1.0]*(self.N-4), 1.5]
    #     diag2 = [*[-0.5]*(self.N-4) ]
    #     return diags([diag2, diag0, diag2], [-2, 0, 2],format="csc")

    # def _mass_inv(self):
    #     return spinv(self._mass())

    # def _stiff(self):
    #     ''' 
    #     Eq. (2.6) of Shen - Effcient Spectral-Galerkin Method II.
        
    #     Equivalent to
    #         S@D2@Si
    #         S, Si : Stencil matrix and its inverse
    #         D2:     Stiffness matrix of the Chebyshev Basis <T''T> 
    #     '''
    #     N,M = self.N-2, self.N-2
    #     D = np.zeros( (N,M) )
    #     for m in range(N):
    #         for n in range(m,M):
    #             if (n==m):
    #                 D[m,n] = -2*(m+1)*(m+2)
    #             elif (n-m)%2==0:
    #                 D[m,n] = -4*(m+1)
    #     return self._to_sparse(D)


    # def stencil(self,inv=False):
    #     '''  
    #     Matrix representation of:
    #         phi_k = T_k - T_{k+2}

    #     See GalerkinChebyshev for details
    #     '''
    #     d0 = [ 1]*(self.N)
    #     d1 = [-1]*(self.N-2)
    #     if inv:
    #         return diags([d0,d1], [0,-2],format="csc")#.toarray()
    #     else:
    #         return diags([d0,d1], [0, 2],format="csc")#.toarray()
