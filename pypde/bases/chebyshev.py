import numpy as np

from scipy.sparse import diags
from .dmsuite import (gauss_lobatto,diff_mat_spectral,
    diff_recurrence_chebyshev,chebdif,pseudoinverse_spectral)
from .fortran import differentiate_cheby
from .memoize import memoized
from .spectralbase import MetaBase
from .utils import *

from scipy.fftpack import dctn as dctn_scipy
import pyfftw

USE_PYFFTW = True

class Chebyshev(MetaBase):
    """
    Function space for Chebyshev polynomials
    .. math::
        T_k = cos(k*arccos(x))
        x_k = cos(pi*k/N); k=0..N

    Parameters:
        N: int
        Number of grid points

    Literature:
    https://www.math.purdue.edu/~shen7/pub/LegendreG.pdf
    https://github.com/spectralDNS/shenfun
    """
    def __init__(self,N,dealias=None):
        x = gauss_lobatto(N-1)
        MetaBase.__init__(self,N,x,dealias)
        self.id = "CH"
        self.family_id = "CH"

        if USE_PYFFTW:
            self.plan_dctn = None # Create on the fly
            self.plan_dctn_shape = None

    def get_basis(self, i=0, x=None):
        if x is None: x = self.x
        w = np.arccos(x)
        return np.cos(i*w)

    def get_basis_derivative(self, i=0, k=0, x=None):
        from numpy.polynomial import chebyshev
        if x is None: x = self.x
        x = np.atleast_1d(x)
        basis = np.zeros(self.N)
        basis[i] = 1
        basis = chebyshev.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        return basis(x)

    def forward_fft(self,f,mass=True):
        '''
        Transform to spectral space via DCT, similar to project(), see
        https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
        Scipys dctn is missing the (-1)^i part, which is handled here
        '''
        sign = np.array([(-1)**k for k in np.arange(self.N)])
        c = 0.5*self.dctn(f)/(self.N-1)
        c = product(sign,c) # mutliplication along first dimension
        return self.solve_mass(c) if mass else c

    def backward_fft(self,c):
        '''  Transform to physical space via DCT '''
        sign = np.array([(-1)**k for k in np.arange(self.N)])
        f = product(sign,c) # mutliplication along first dimension
        f[[0,-1]] = product(np.array([2,2]),f[[0,-1]]) # first and last times 2
        return 0.5*self.dctn(f)

    def dctn(self,f,axes=(0,),use_pyfftw=USE_PYFFTW):
        ''' '''
        if use_pyfftw:
            if self.plan_dctn is None or f.shape != self.plan_dctn_shape:
                self.plan_dctn = self.create_plan_dctn(f.shape,axes)
                self.plan_dctn_shape = f.shape
            return self.plan_dctn(f)
        else:
            return dctn_scipy(f,type=1,axes=axes)

    #@memoized
    def create_plan_dctn(self,shape,axes=(0,)):
        a = pyfftw.empty_aligned(shape, dtype='float64')
        b = pyfftw.empty_aligned(shape, dtype='float64')
        return pyfftw.FFTW(a, b, axes=axes, direction='FFTW_REDFT00')

    @memoized
    def dmp_collocation(self,deriv):
        ''' Collocation derivative matrix, acts in physical space.'''
        return chebdif(self.N,deriv)[1]

    @memoized
    def dms(self,deriv):
        '''
        Chebyshev differentation matrix, acts in spectral space.
        Differentiation can be done more efficientrly via recurrence,
        see self.derivative

                self.stiff = self.mass @ self.dms(2)
        '''
        return diff_mat_spectral(self.N,deriv)

    def derivative(self,fhat,deriv,out_cheby=True):
        if deriv == 0:
            return fhat
        #return diff_recurrence_chebyshev(fhat,deriv)
        df = fhat
        if fhat.ndim == 1:
            for i in range(deriv):
                df = differentiate_cheby.diff_1d(df)
            return df
        elif fhat.ndim == 2:
            for i in range(deriv):
                df = differentiate_cheby.diff_2d(df)
            return df
        else:
            raise NotImplementedError("ndim must be less than 3")


    def derivative_physical(self,f,deriv,method="fft"):
        assert method in ["fft","spectral","dm","physical"]
        ''' Calculate derivative of input array f'''
        if method in ("fft", "spectral"):
            c = self.forward_fft(f)
            dc = self.derivative(c,deriv)
            return self.backward_fft(dc)
        elif method in ("dm", "physical"):
            return self.dmp_collocation(deriv)@f

    def solve_mass(self,f):
        '''
        Solve Mx = f, where m is mass matrix.
        In this case M is purely diagonal.
        '''
        m_inv = diags([1.0, *[2.0]*(self.N-2), 1.0],0)
        return m_inv@f

    # ---------------------------------------------
    #    Pseudoinverse
    # ---------------------------------------------

    def B(self,deriv,discardrow=0):
        ''' Pseudoinverse '''
        from .pinv import pseudo_inv
        if deriv>2:
            raise ValueError("deriv>2 not supported")
        return pseudo_inv(self,D = deriv)[discardrow:,:]#self.mass_inv

    def I(self,discardrow=0):
        ''' (Discarded) Identitiy matrix '''
        return np.eye(self.N)[discardrow:,:]


class GalerkinChebyshev(MetaBase):
    """
    Base class for composite Chebyshev spaces
    like ChebDirichlet used in the Galerkin method.

    Define linear combination of Chebyhsev polynomials
    in stencil and set size by slice.

    """
    def __init__(self,N,dealias=None):
        x = gauss_lobatto(N-1)
        MetaBase.__init__(self,N,x,dealias)

        # Boundary conditions
        self._bc = None
        self._coeff_bc = None

        # Get family info
        self.family_id = "CH"
        self.family = Chebyshev(self.N)

    def get_basis(self,i=0,x=None):
        return self.get_basis_derivative(i=i,k=0,x=x)

    def get_basis_derivative(self, i=0, k=0, x=None):
        from numpy.polynomial import chebyshev
        if x is None: x = self.x
        x = np.atleast_1d(x)
        if i < self.M:
            basis = self.stencil()[:,i]
            basis = chebyshev.Chebyshev(basis)
            if k > 0:
                basis = basis.deriv(k)
            return basis(x)
        else:
            raise ValueError("basis not known for i={:4d}"
                .format(i))

    def slice(self):
        '''
        Galerkin space usually of size [0,N-3] (+ 2 BCs bases)
        Can be overwritten in child classes
        '''
        return slice(0, self.N-2)

    def _stencil(self):
        '''
        Must be implemented on child class!
        All Transformations are derived from the stencil.

        Stencil Matrix (NxM) to transform :
            Galerkin (M) -> Chebyshev (N)
                    u = S v

        Stencil:
            N x M Matrix

        Literature:
            K. Julien: doi:10.1016/j.jcp.2008.10.043
        '''
        raise NotImplementedError

    def stencil(self,transpose=False):
        if transpose:
            return self._stencil().T
        return self._stencil()

    # ---------------------------------------------
    #      Forward & Backward transform
    # ---------------------------------------------

    def forward_fft(self,f,bc=None):
        '''
        Transform to spectral space via DCT
        Applied along zero axis of f

             S^t T u = S^t@M@S vhat
        -->  S^t T u = Mv vhat

            T u: Chebyshev transform without mass_inv applied
            M : Mass Chebyshev
            Mv: Mass Galerkin (diag -2,0,2)
            S : Transform Stencil

        Input
            f: N x M array
                Array in real space
            bc: 2 x M array (optional)
                galerkin coefficients of BCs
        '''
        #  Subtract inhomogeneous part
        if bc is not None:
            f = f - self.eval_inhomogeneous(bc)

        c = self.family.forward_fft(f,mass=True)

        return self.from_chebyshev(c)


    def backward_fft(self,c,bc=None):
        '''
        Transform to physical space via DCT
        Applied along zero axis of c

        Input
            c: N x M array
                galerkin coefficients
            bc: 2 x M array (optional)
                galerkin coefficients of BCs
        '''
        c = self.to_chebyshev(c)

        # Add inhomogeneous part back
        if bc is not None:
            c += self.bc.to_chebyshev(bc)

        return self.family.backward_fft(c)

    def to_chebyshev(self,vhat):
        '''
        Transform form galerkin to chebyshev
                    S v = u
        '''
        assert vhat.shape[0] == self.M,"{} {}".format(vhat.shape[0],self.M)
        return self.S_sp@vhat

    def from_chebyshev(self,uhat):
        '''
        Transform form chebyshev to galerkin
                    S v = u
        '''
        assert uhat.shape[0] == self.N,\
        "Shape mismatch ({:3}) ({:3})".format(uhat.shape[0],self.N)
        return self._solve_stencil_inv(uhat)

    def _solve_stencil_inv(self,uhat):
        '''
        This function works only if the stencil
        is of size N x N-2 and populated on
        the diagonals 0 and -2
        Original system is overdetermine, hence
        multiply with S^T

               S^T S v = S^T u

        Obtain galerkin coefficients v (N -2 x M)
        from chebyshev coefficients u (N x M)

        Note:
            For a generic stencil S, one possibility
            would is np.linalg.lstsq(S,uhat)
        '''
        #from .linalg.tdma import TDMA_offset as TDMA
        from .linalg.tdma import TDMA_Fortran as TDMA
        l2,d,u2=self._init_stencil_inv()
        #try:
        return TDMA(l2,d,u2,self.ST_sp@uhat,2)
        #except:
        #    return np.linalg.lstsq(self.S,uhat)[0]

    @memoized
    def _init_stencil_inv(self):
        from .utils import extract_diag
        ''' Return diagonals -2,0,2 of S.T @ S'''
        A = self.S.T@self.S
        return extract_diag(A,k=(-2,0,2))

    def eval_inhomogeneous(self,bchat):
        '''
        Calculates the inhomogeneous part of f
            f = f_h + f_i
        for given boundary coefficients
        '''
        return self.bc.backward_fft(bchat)

    def derivative(self,vhat,deriv,out_cheby=True):
        '''
        Input
            vhat ndarray
                Galerkin coefficients
            deriv int

        Return
            Coefficient array of derivative,
            returns chebyshev coeff if out_cheby is True,,
            otherwise return galerkin coeff
        '''
        uhat = self.to_chebyshev(vhat)
        duhat =  self.family.derivative(uhat,deriv)
        if out_cheby:
            return duhat
        return self.from_chebyshev(duhat)


class ChebDirichlet(GalerkinChebyshev):
    """
    Function space for Dirichlet boundary conditions
    .. math::
        \phi_k = T_k - T_{k+2}

    Parameters:
        N: int
            Number of grid points
    """
    def __init__(self,N,dealias=None):
        GalerkinChebyshev.__init__(self,N,dealias)
        self.id = "CD"
        self.bc = DirichletC(N)

    def _stencil(self):
        '''
        Matrix representation of:
            phi_k = T_k - T_{k+2}

        See GalerkinChebyshev class
        '''
        S = np.zeros((self.N,self.M))
        for i in range(self.M):
            S[i,i],S[i+2,i] = 1, -1
        return S

class ChebNeumann(GalerkinChebyshev):
    """
    Function space for Neumann boundary conditions
    .. math::
        \phi_k = T_k - k^2/(k+2)^2 T_{k+2}

    Parameters:
        N: int
            Number of grid points
    """
    def __init__(self,N,dealias=None):
        GalerkinChebyshev.__init__(self,N,dealias)
        self.id = "CN"
        self.bc = NeumannC(N)

    def _stencil(self):
        '''
        Matrix representation of:
            phi_k = T_k - T_{k+2}

        See GalerkinChebyshev class
        '''
        S = np.zeros((self.N,self.M))
        for i in range(self.M):
            S[i,i],S[i+2,i] = 1, -(i/(i+2))**2
        return S



# ---------------------------------------------
#       Bases for Boundary Conditions
# ---------------------------------------------

class DirichletC(GalerkinChebyshev):
    """
    Function space purely for Dirichlet boundary conditions
    .. math::
        \phi_0 = 0.5*T_0 - 0.5*T_1
        \phi_1 = 0.5*T_0 + 0.5*T_1

    Parameters:
        N: int
            Number of grid points
    """
    def __init__(self,N,dealias=None):
        GalerkinChebyshev.__init__(self,N,dealias)
        self.id = "DC"
        self.is_bc = True

    def _stencil(self):
        S = np.zeros((self.N,self.M))
        S[0,0],S[1,0] =  0.5,-0.5
        S[0,1],S[1,1] =  0.5, 0.5
        return S

    def slice(self):
        return slice(0,2)


    def from_chebyshev(self,c):
        ''' DirichletC only depends on first two modes '''
        return np.linalg.solve(self.S[:2,:2],c[:2])

class NeumannC(GalerkinChebyshev):
    """
    Function space purely for Neumann boundary conditions
    .. math::
        \phi_N-2 = 0.5*T_0 - 1/8*T_1
        \phi_N-1 = 0.5*T_0 + 1/8*T_1

    Parameters:
        N: int
            Number of grid points
    """
    def __init__(self,N,dealias=None):
        GalerkinChebyshev.__init__(self,N,dealias)
        self.id = "NC"
        self.is_bc = True

    def _stencil(self):
        S = np.zeros((self.N,self.M))
        S[0,0],S[1,0] =  0.5,-1/8
        S[0,1],S[1,1] =  0.5, 1/8
        return S

    def slice(self):
        return slice(0,2)

    def from_chebyshev(self,c):
        ''' NeumannC only depends on first two modes '''
        return np.linalg.solve(self.S[:2,:2],c[:2])
