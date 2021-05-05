import numpy as np
from scipy.fftpack import dctn
from scipy.sparse import diags
from .dmsuite import (gauss_lobatto,diff_mat_spectral,
    diff_recurrence_chebyshev,chebdif,pseudoinverse_spectral)
from .memoize import memoized
from .spectralbase import MetaBase
from .utils import *

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
    def __init__(self,N):
        x = gauss_lobatto(N-1)
        MetaBase.__init__(self,N,x)
        self.id = "CH"
        self.family_id = "CH"

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
        c = 0.5*dctn(f,type=1,axes=0)/(self.N-1)
        c = product(sign,c) # mutliplication along first dimension
        return self.solve_mass(c) if mass else c

    def backward_fft(self,c):
        '''  Transform to physical space via DCT '''
        sign = np.array([(-1)**k for k in np.arange(self.N)])
        f = product(sign,c) # mutliplication along first dimension
        f[[0,-1]] = product(np.array([2,2]),f[[0,-1]]) # first and last times 2
        return 0.5*dctn(f,type=1,axes=0)

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

    def derivative(self,fhat,deriv):
        if deriv == 0:
            return fhat
        return diff_recurrence_chebyshev(fhat,deriv)

    def derivative_physical(self,f,deriv,method="fft"):
        assert method in ["fft","spectral","dm","physical"]
        ''' Calculate derivative of input array f'''
        if method in ("fft", "spectral"):
            c = self.forward_fft(f)
            dc = self.derivative(c,deriv)
            return self.backward_fft(dc)
        elif method in ("dm", "physical"):
            return self.dmp_collocation(deriv)@f

    def _mass_inv(self):
        return self.inner_inv(D = 0 )

    def _grad_inv(self):
        return self.inner_inv(D = 1 )

    def _stiff_inv(self):
        return self.inner_inv(D = 2 )

    def solve_mass(self,f):
        '''
        Solve Mx = f, where m is mass matrix.
        In this case M is purely diagonal.
        '''
        m_inv = diags([1.0, *[2.0]*(self.N-2), 1.0],0)
        return m_inv@f


class GalerkinChebyshev(MetaBase):
    """
    Base class for composite Chebyshev spaces
    like ChebDirichlet used in the Galerkin method.

    Define linear combination of Chebyhsev polynomials
    in stencil and set size by slice.

    """
    def __init__(self,N):
        x = gauss_lobatto(N-1)
        MetaBase.__init__(self,N,x)

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

    @property
    def S(self):
        ''' Shortcut for stencil'''
        return self.stencil()
    
    @property
    def S_sp(self):
        ''' Sparse version of stencil '''
        return self._tosparse( self.stencil() )

    @property
    def ST_sp(self):
        ''' Sparse version of stencil '''
        return self._tosparse( self.stencil(transpose=True) )

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
        assert uhat.shape[0] == self.N
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
        from .linalg.tdma import TDMA_offset as TDMA
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
    def __init__(self,N):
        GalerkinChebyshev.__init__(self,N)
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
    def __init__(self,N):
        GalerkinChebyshev.__init__(self,N)
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
    def __init__(self,N):
        GalerkinChebyshev.__init__(self,N)
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
    def __init__(self,N):
        GalerkinChebyshev.__init__(self,N)
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




    # ---------------------------------------------
    #           Old Stuff
    # ---------------------------------------------


    # def _stencilbc(self):
    #     '''
    #     Should be implemented on child class for
    #     inhomogenoeus BCs!

    #     Stencil transforms boundary coefficients (phi_N-1 & phi_N)
    #     to chebyshev coefficients

    #     Stencilbc:
    #         N x 2 Matrix
    #     '''
    #     raise NotImplementedError

    # def stencilbc(self,transpose=False):
    #     if transpose:
    #         return self._stencilbc().T
    #     return self._stencilbc()

    # def stencilfull(self):
    #     return np.hstack( (self.stencil(),self.stencilbc()) )
        #return inner(self,self.bc)

    # def _stencilbc(self):
    #     '''
    #         phi_N-1 = 0.5*(T_0-T_1)
    #         phi_N-2 = 0.5*(T_0+T_1)
    #     '''
    #     S = np.zeros((self.N,2))
    #     S[0,0],S[0,1] =  0.5, 0.5
    #     S[1,0],S[1,1] = -0.5, 0.5
    #     return S

    # def _stencilbc(self,inv=True):
    #     '''
    #         phi_N-1 = 1/2*T_1-1/8*T_2
    #         phi_N-2 = 1/2*T_1+1/8*T_2
    #     '''
    #     S = np.zeros((2,self.N))
    #     S[0,0],S[0,1] =  0.5, 0.5
    #     S[1,0],S[1,1] = -1/8, 1/8
    #     if inv: return S.T
    #     return S





    # def _to_galerkin(self,cheby_c):
    #     '''
    #     Transform from T to phi basis

    #     cheby_c are the chebyshev coefficients
    #     before chebyshev.solve_mass(c), i.e. cheby_c = M@c
    #     '''
    #     c = self.stencil(True)@cheby_c

    #     return self.solve_mass(c)

    # def _to_chebyshev(self,galerkin_c):
    #     '''
    #     Transform from phi to T basis
    #               u = S v
    #     '''
    #     return self.stencil()@galerkin_c

    # @memoized
    # def _init_solve_mass(self):
    #     from .utils import extract_diag
    #     ''' Return diagonals -2,0,2 of mass matrix'''
    #     return extract_diag(self.mass,k=(-2,0,2))

    # def solve_mass(self,f):
    #     from .linalg.tdma import TDMA_offset as TDMA
    #     '''
    #     Solve Mx = f, where m is the mass matrix.
    #     M has entries on -2,0,2, which can be solved
    #     efficiently

    #     Note:
    #         Not generic for all combined galerkin bases,
    #         keep in mind.
    #     '''
    #     l2,d,u2=self._init_solve_mass()
    #     try:
    #         return TDMA(l2,d,u2,f,2)
    #     except:
    #         return np.linalg.solve(self.mass,f)

    # ---------------------------------------------
    #      Handle boundary conditions
    # ---------------------------------------------

    # @memoized
    # def _bcmat(self):
    #     '''  equivalent to S@inner(self,self.bc) '''
    #     return self.family.mass@self.bc.stencil()

    # def _add_bc_after_to_chebyshev(self,cheby_c, bc):
    #     '''
    #     Add bc coefficients to chebyshev coefficients
    #     '''
    #     cheby_c += self.bc._to_chebyshev(bc)

    # def _subtract_bc_before_to_galerkin(self,cheby_c, bc):
    #     '''
    #     When transforming from Chebyshev -> Galerkin coefficients
    #     the contributions of the bc_coefficients must be subtracted
    #     from the chebyshev coefficients.

    #         S^T @ ( cheby_c - M_bc @ bc ) =  M @ galerkin_c

    #     Apply before .solve_mass
    #     '''
    #     assert bc.shape[0] == self._bcmat().shape[1], \
    #     "First dimension of bc must be {:1d}".format(self._bcmat().shape[1])
    #     cheby_c -= self._bcmat()@bc





    # @memoized
    # def pseudoinverse(self,deriv,discardrow=None):
    #     '''
    #     Pseudoinverse of differentiation matrix dms.
    #     Makes system banded and fast to solve.
    #     Returns pseudoidentity matrix if deriv is zero
    #     '''
    #     assert deriv==2 or deriv==1 or deriv==0, \
    #     "Only deriv==1,2 or 0 supported"
    #     if discardrow is None: discardrow = deriv

    #     if deriv==0:
    #         rv = np.eye(self.N); rv[0,0] = rv[1,1] = 0
    #     elif deriv==1 or deriv==2:
    #         rv = pseudoinverse_spectral(self.N,deriv)

    #     return rv[discardrow:,:]


# def twodma2(l,d,b):

#     '''
#     Solve
#         Ax = b
#     A (N x N-2) is banded with diagonals in offsets -2, 0
    
#     Input
#         l,d: 1d arrays
#             diagonals -2, 0
#         b: nd array (N)

#     Return
#         array (N-2)
#     '''
#     n = b.shape[0]
#     x = np.zeros(b.shape)[2:]
#     x[0] = b[0]/d[0]
#     x[1] = b[1]/d[1]
#     for i in range(2,n-2):
#         x[i] = (b[i] - x[i-2]*l[i-2])/d[i]
#     print(b.shape)
#     print(l.shape)
#     x[n-4] = b[n-2]/l[n-5]
#     x[n-3] = b[n-1]/l[n-5]

#     return x