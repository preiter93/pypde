import numpy as np
from scipy.sparse import diags
from .utils import tosparse
from itertools import tee
import warnings

def inner(u,v,w="GL",D=(0,0),**kwargs):
    import pypde.bases.spectralbase as sb
    '''
    Inner product of <u,v>_w, where w is the weight.

    The function can handle 3 scenarios:
    1)  u: array
        v: array
    2)  u: MetaBase (over basis or derivative of basis)
        v: array
    3)  u: MetaBase
        v: MetaBase


    Input
        u: ndarray or MetaBase
        v: ndarray or MetaBase
        w: ndarray or "GL"
            Weights of inner product
            "GL": Gauss-Lobatto weight
            "NO": No weighs, use array of ones
        D: tuple of integers, size 1 or 2
            Order of derivative if input is a MetaBase
            (0: Standard Basis )

    Output
        (1) scalar, (2) array or (3) matrix
    '''
    # -- check inputs
    if all([isinstance(i,np.ndarray) for i in [u,v]]):
        assert u.size==v.size
        method = "scalar"
        N = u.size
    elif isinstance(u,sb.MetaBase) and isinstance(v,np.ndarray):
        assert u.N==v.size #or u.M==v.size
        method = "array"
        N = v.size
    elif isinstance(v,sb.MetaBase) and isinstance(u,np.ndarray):
         return inner(v,u,w,D,**kwargs) # Switch u and v
    elif all([isinstance(i,sb.MetaBase) for i in [u,v]]):
        assert u.N==v.N
        method = "matrix"
        N = u.N
    else:
        raise ValueError("u and v must be of type ndarray or MetaBase")

    # -- handle weights
    assert w in ["GL", "NO"], "Unknown weight key supplied. Known are 'NO','GL'."
    if w == ("GL"):
        w_ = np.concatenate(([0.5],np.ones(N-2),[0.5] ))
    elif w == ("NO"):
        w_ = np.ones(N)

    # -- inner product
    if method=="scalar":
        return np.sum(u*v*w_)/(N-1)

    if method=="array":
        return np.array([inner(uu,v,w) for uu in u.iter_basis()])

    if method=="matrix":
        return _inner_spectralbase(u,v,w,D,**kwargs)

def _inner_spectralbase(u,v,w,D,lookup=True):
    '''
    Calculates inner product of two SpectralBases, should be
    called from inner
    '''
    assert len(D)==2 or len(D)==1
    if len(D)==2: ku,kv=D
    if len(D)==1: ku=kv=D

    # Check list of known inner products first
    if lookup:
        value = InnerKnown().check(u,v,ku,kv)
        if value is not None:
            return value

    # Get Generators from MetaBase
    gen1 = u.iter_basis() if ku==0 else u.iter_deriv(k=ku)
    gen2 = v.iter_basis() if kv==0 else v.iter_deriv(k=kv)

    M = []
    for j,uu in enumerate(gen1):
        gen2,gen2_clone = tee(gen2) # clone iterator
        c = [inner(uu,vv,w) for vv in gen2_clone]
        M.append(c)

    return np.vstack(M)


class InnerKnown():
    '''
    This class contains familiar inner products of the Functionspaces
    and its derivative.

    Use:
    InnerKnown().check(u,v,ku,kv)

    If the two input Functionspaces u,v are of the same type, class
    looks up if the key is contained in dict, and if so returns
    the stored value,otherwise gives it back to inner to calculate it.

    Example:
            Inner product <TiTj> of Chebyshev polynomials T (=mass matrix)
            has the entries [1,0.5,...,0.5,1], this is stored under the key
            'CH^0,CH^0'

    If the two input Functionspaces are of different type, but one
    is the family space (T) of the other (P), class tries to derive the inner
    product from the familys inner product and its childs  stencil .
        <P,T> = S@<T,T>
    '''
    inverted = False
    @property
    def dict(self):
        return{
            # <Chebyshev Chebyshev>
            "CH^0,CH^0": self.chebyshev_mass,
            "CH^0,CH^1": self.chebyshev_grad,
            "CH^0,CH^2": self.chebyshev_stiff,
            "CH^0,CH^3": self.chebyshev_cube,
            "CH^0,CH^4": self.chebyshev_quad,

            # <ChebDirichlet ChebDirichlet>
            "CD^0,CD^0": self.chebdirichlet_mass,
            "CD^0,CD^1": self.chebdirichlet_grad,
            "CD^0,CD^2": self.chebdirichlet_stiff,
            "CD^0,CD^3": self.chebdirichlet_cube,
            "CD^0,CD^4": self.chebdirichlet_quad,

            # <ChebDirichlet ChebDirichlet>
            "CN^0,CN^0": self.chebneumann_mass,
            "CN^0,CN^1": self.chebneumann_grad,
            "CN^0,CN^2": self.chebneumann_stiff,
            "CN^0,CN^3": self.chebdirichlet_cube,
            "CN^0,CN^4": self.chebdirichlet_quad,

            # <Fourier Fourier>
            "FO^0,FO^0": self.fourier_mass,
            "FO^0,FO^1": self.fourier_grad,
            "FO^0,FO^2": self.fourier_stiff,
            "FO^0,FO^3": self.fourier_cube,
            "FO^0,FO^4": self.fourier_quad,
        }

    def check(self,u,v,ku,kv):
        '''
        See class' explanation.

        Input:
            u, v:  MetaBase
            ku,kv: Integers (Order of derivative)
        '''
        assert all(hasattr(i,"id") for i in [u,v])
        assert u.family_id == v.family_id, "Test and Trial should be of same family.(?)"

        # Figure out combination <parent base,child_base>, etc
        generation_u = "parent" if u.id == u.family_id else "child"
        generation_v = "parent" if v.id == v.family_id else "child"

        # Put higher derivative in the end, so that key is unique
        # and transpose later
        if ku>kv:
            ku,kv  = kv,ku
            self.inverted = True
        # Generate Key
        key_family = self.generate_key(u.family_id,v.family_id,ku,kv)
        key        = self.generate_key(u.id,v.id,ku,kv)

        # Lookup Key
        if key in self.dict:
            #  Inner product of input matches is known
            value = self.dict[key](u=u,v=v,ku=ku,kv=kv)
            if self.inverted: value = value.T
            return value
        elif key_family in self.dict:
            #  Inner product of family is known from which childs inner can be derived
            value = self.dict[key_family](u=u,v=v,ku=ku,kv=kv)
            if self.inverted: value = value.T
        else:
            warnings.warn("Key or Family key {:s} not found. Derive from family...".
                format(key))
            return None

        # Derive inner product from parent (T) with transform stencil S
        if [generation_u,generation_v] == ["parent", "child"]:
            value = value@tosparse(v.stencil())

        if [generation_u,generation_v] == ["child", "parent"]:
            value = tosparse(u.stencil().T)@value

        if [generation_u,generation_v] == ["child", "child"]:
            value = u.stencil().T@value@v.stencil()

        return value

    def generate_key(self,idu,idv,ku,kv):
        return "{:2s}^{:1d},{:2s}^{:1d}".format(idu,ku,idv,kv)

    # ----- Collection of known inner products ------
    @staticmethod
    def chebyshev_mass(u=None,**kwargs):
        ''' <Ti Tj>
        Some literature gives it with a factor of pi
        '''
        #assert u.id=="CH"
        return diags([1.0, *[0.5]*(u.N-2), 1.0],0).toarray()

    #@staticmethod
    def chebyshev_grad(self,u=None,k=1,**kwargs):
        '''  <Ti Tj^1> (todo: find an analytical expression)'''
        from .dmsuite import diff_mat_spectral as dms
        mass = self.chebyshev_mass(u)
        return tosparse( mass@dms(u.N,k) ).toarray()

    #@staticmethod
    def chebyshev_stiff(self,u=None,**kwargs):
        '''  <Ti Tj^2> '''
        return self.chebyshev_grad(u,2)

    def chebyshev_cube(self,u=None,**kwargs):
        '''  <Ti Tj^3> '''
        return self.chebyshev_grad(u,3)

    def chebyshev_quad(self,u=None,**kwargs):
        '''  <Ti Tj^4> '''
        return self.chebyshev_grad(u,4)

    @staticmethod
    def chebdirichlet_mass(u=None,**kwargs):
        '''
        <Phi Phi>
        Eq. (2.5) of Shen - Effcient Spectral-Galerkin Method II.
        '''
        diag0 = [1.5, *[1.0]*(u.N-4), 1.5]
        diag2 = [*[-0.5]*(u.N-4) ]
        return diags([diag2, diag0, diag2], [-2, 0, 2]).toarray()

    @staticmethod
    def chebdirichlet_grad(u=None,**kwargs):
        '''
        <Phi Phi^1>
        Eq. (4.6) of Shen - Effcient Spectral-Galerkin Method II.
        '''
        diag0,diag1 = -np.arange(2,u.N-1),np.arange(1,u.N-2)
        return diags([diag0, diag1], [-1, 1]).toarray()
        #  -- Alternative ---
        # S,Si = u.stencil(), u.stencil(inv=True)
        # D1 = self.chebyshev_grad(u)
        # return S@D1@Si
        # -------------------

    @staticmethod
    def chebdirichlet_stiff(u=None,**kwargs):
        '''
         <Phi Phi^2>
        Eq. (2.6) of Shen - Effcient Spectral-Galerkin Method II.

        Equivalent to
            S@D2@Si
            S, Si : Stencil matrix T<->Phi and its transpose
            D2:     Stiffness matrix of the Chebyshev Basis <T''T>
        '''
        N,M = u.M,u.M
        D = np.zeros( (N,M) )
        for m in range(N):
            for n in range(m,M):
                if (n==m):
                    D[m,n] = -2*(m+1)*(m+2)
                elif (n-m)%2==0:
                    D[m,n] = -4*(m+1)
        return tosparse(D).toarray()

    def chebdirichlet_cube(self,u=None,**kwargs):
        '''  <Phi Phi^3> '''
        D = self.chebyshev_cube(u)
        return u.stencil().T@D@u.stencil()

    def chebdirichlet_quad(self,u=None,**kwargs):
        '''  <Phi Phi^4> '''
        D = self.chebyshev_quad(u)
        return u.stencil().T@D@u.stencil()


    @staticmethod
    def chebneumann_mass(u=None,**kwargs):
        '''
        <Phi Phi>
        See 8.2.2 Neumann
            AMS Kruseman - A Chebyshev-Galerkin method for inertial waves
        '''
        cN = np.array([1.0, *[0.5]*(u.N-2),1.0])
        b  = np.array([(i/(i+2))**2 for i in range(u.N-2)])
        diag0 = (1*cN[:-2]+b**2*cN[2:])
        diag2 = -b[:-2]*cN[:-4]
        return diags([diag2, diag0, diag2], [-2, 0, 2]).toarray()

    def chebneumann_grad(self,u=None,**kwargs):
        '''
        <Phi Phi^1>
        '''
        D = self.chebyshev_grad(u)
        return u.stencil().T@D@u.stencil()

    @staticmethod
    def chebneumann_stiff(u=None,**kwargs):
        '''
         <Phi Phi^2>
        '''
        N,M = u.M,u.M
        D = np.zeros( (N,M) )
        for m in range(N):
            for n in range(m,M):
                if (n==m):
                    D[m,n] = -2*m**2*(m+1)/(m+2)
                elif (n-m)%2==0:
                    D[m,n] = -4*n**2*(m+1)/(m+2)**2
        return tosparse(D).toarray()


    @staticmethod
    def fourier(u,k):
        return diags(0.5*u._k**k,0).toarray()

    def fourier_mass(self,u=None,**kwargs):
        ''' <Fi Fj> '''
        return self.fourier(u=u,k=0)

    def fourier_grad(self,u=None,**kwargs):
        ''' <Fi Fj'> '''
        return self.fourier(u=u,k=1)

    def fourier_stiff(self,u=None,**kwargs):
        ''' <Fi Fj^2> '''
        return self.fourier(u=u,k=2)

    def fourier_cube(self,u=None,**kwargs):
        ''' <Fi Fj^3> '''
        return self.fourier(u=u,k=3)

    def fourier_quad(self,u=None,**kwargs):
        ''' <Fi Fj^4> '''
        return self.fourier(u=u,k=4)


def inner_inv(u,D=0,**kwargs):
    import pypde.bases.spectralbase as sb
    '''
    Returns Pseudoinverse of inner product of <u,u^D>_w.

    At the moment the inverse is only used to make
    chebyshev system banded and efficient.

    NOTE:
        Only supports Chebyshev Bases "CH" and the
        derivatives D=(0,1) and (0,2)

    Input
        u:  MetaBase
        D:  int
            Order of derivative

    Output
        matrix
    '''
    assert isinstance(u,sb.MetaBase)
    return InnerInvKnown().check(u,D)



class InnerInvKnown():
    '''
    This class contains familiar inversese of inner products

    Use:
    InnerInvKnown().check(u,v,ku,kv)

    '''
    inverted = False
    @property
    def dict(self):
        return{
            # <Chebyshev Chebyshev>
            "CH^0,CH^0": self.chebyshev_mass_inv,
            "CH^0,CH^1": self.chebyshev_grad_inv,
            "CH^0,CH^2": self.chebyshev_stiff_inv,
        }

    def check(self,u,ku):
        '''
        Input:
            u:  MetaBase
            ku: Integers (Order of derivative)
        '''
        assert all(hasattr(i,"id") for i in [u])
        assert u.id == "CH", "InnerInv supports only Chebyshev at the moment"
        assert ku == 0 or ku == 1 or ku == 2

        key = self.generate_key(u.id,u.id,0,ku)

        # Lookup Key
        if key in self.dict:
            value = self.dict[key](u=u,ku=ku)
            return value
        else:
            raise ValueError("Key not found in inner_inv.")

    def generate_key(self,idu,idv,ku,kv):
        idv, ku = idu, 0
        return "{:2s}^{:1d},{:2s}^{:1d}".format(idu,ku,idv,kv)

    # ----- Collection of known inverses of inner products ------
    @staticmethod
    def chebyshev_mass_inv(u=None,**kwargs):
        ''' <Ti Tj>^-1
        Some literature gives it with a factor of pi
        '''
        #assert u.id=="CH"
        return diags([1.0, *[2.0]*(u.N-2), 1.0],0).toarray()

    #@staticmethod
    def chebyshev_grad_inv(self,u=None,k=1,**kwargs):
        '''
        Pseudoinverse
        <Ti Tj^1>^-1
        First row can be discarded
        '''
        from .dmsuite import pseudoinverse_spectral as pis
        mass_inv = self.chebyshev_mass_inv(u)
        return tosparse( pis(u.N,k)@mass_inv ).toarray()

    def chebyshev_stiff_inv(self,u=None,**kwargs):
        '''
        Pseudoinverse:
        <Ti Tj^2>^-1
        First two rows can be discarded
         '''
        return self.chebyshev_grad_inv(u,2)
