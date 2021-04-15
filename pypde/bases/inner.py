import numpy as np
from scipy.sparse import diags
from .utils import to_sparse
from itertools import tee
import warnings

def inner(u,v,w="GL",D=(0,0),**kwargs):
    import pypde.bases.spectralbase as sb
    '''
    Inner product of <u,v>_w, where w is the weight.
    
    The function can handle 3 scenarios:
    1)  u: array
        v: array
    2)  u: SpectralBase (over basis or derivative of basis)
        v: array
    3)  u: SpectralBase
        v: SpectralBase
    
    
    Input
        u: ndarray or SpectralBase
        v: ndarray or SpectralBase
        w: ndarray or "GL" 
            Weights of inner product 
            "GL": Gauss-Lobatto weight
            "NO": No weighs, use array of ones
        D: tuple of integers, size 1 or 2
            Order of derivative if input is a SpectralBase
            (0: Standard Basis )
        
    Output
        (1) scalar, (2) array or (3) matrix
    '''
    # -- check inputs
    if all([isinstance(i,np.ndarray) for i in [u,v]]):
        assert u.size==v.size
        method = "scalar"
        N = u.size
    elif isinstance(u,sb.SpectralBase) and isinstance(v,np.ndarray):
        assert u.N==v.size #or u.M==v.size
        method = "array"
        N = v.size
    elif isinstance(v,sb.SpectralBase) and isinstance(u,np.ndarray):
         return inner(v,u,w,D,**kwargs) # Switch u and v    
    elif all([isinstance(i,sb.SpectralBase) for i in [u,v]]):
        assert u.N==v.N
        method = "matrix"
        N = u.N
    else:
        raise ValueError("u and v must be of type ndarray or SpectralBase")

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

    # Get Generators from Spectralbase
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
        }

    def check(self,u,v,ku,kv):
        ''' 
        See class' explanation. 

        Input:
            u, v:  SpectralBase
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
            warnings.warn("Key or Family key {:s} not found. Find Inner() numerically...".
                format(key))
            return None

        # Derive inner product from parent (T) with transoform stencil S
        if [generation_u,generation_v] == ["parent", "child"]:
            value = value@to_sparse(v.stencil(inv=True))
            
        if [generation_u,generation_v] == ["child", "parent"]:
            value = to_sparse(u.stencil())@value

        if [generation_u,generation_v] == ["child", "child"]:
            value = to_sparse(u.stencil())@value@to_sparse(v.stencil(inv=True))

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
        return to_sparse( mass@dms(u.N,k) ).toarray()

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

    #@staticmethod
    def chebdirichlet_grad(self,u=None,**kwargs):
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
        return to_sparse(D).toarray()

    def chebdirichlet_cube(self,u=None,**kwargs):
        '''  <Phi Phi^3> '''
        S,Si = u.stencil(), u.stencil(inv=True) 
        D = self.chebyshev_cube(u)
        return S@D@Si

    def chebdirichlet_quad(self,u=None,**kwargs):
        '''  <Phi Phi^3> '''
        S,Si = u.stencil(), u.stencil(inv=True) 
        D = self.chebyshev_quad(u)
        return S@D@Si

    




# def _known_inner_products(u,v,ku,kv):
#     ''' 
#     Collection of exact inner products of Functionspaces. 
#     This should be called before _inner_spectralbase calculates 
#     the inner products numerically.

#     Keys consist of FunctionSpace ID's and the derivatives;
#     they are stored in the dict _list_known_inner_products

#     Example:
#         Inner product <TiTj> of Chebyshev polynomials T (=mass matrix)
#         has the entries [1,0.5,...,0.5,1], this is stored under the key
#         'CH^0,CH^0'
#     '''
#     assert all(hasattr(i,"id") for i in [u,v])

#     # Put higher derivative in the end, so that key is unique
#     if ku>kv:
#         u, v  = v, u
#         ku,kv = kv,ku
#     # Generate Key
#     key = "{:2s}^{:1d},{:2s}^{:1d}".format(u.id,ku,v.id,kv)

#     # Lookup Key
#     if key in _list_of_known_inner_products:
#         print("Key {:} exists. Use lookup value.".format(key))
#         value = _list_of_known_inner_products[key](u=u,v=v,ku=ku,kv=kv)
#         return True, value

#     # Key not found. Add to  inner product is known analytically
#     warnings.warn("Key {:} not found in list. Use inner() instead.".format(key))
#     return False, None


# def chebyshev_mass(u=None,**kwargs):
#     ''' <Ti Tj>
#     Some literature gives it with a factor of pi
#     '''
#     assert u.id=="CH"
#     return diags([1.0, *[0.5]*(u.N-2), 1.0],0).toarray()

# def chebyshev_grad(u=None,k=1,**kwargs):
#     '''  <Ti' Tj> (todo: find an analytical expression)'''
#     from .chebyshev import diff_mat_spectral as dms 
#     mass = chebyshev_mass(u)
#     return to_sparse( mass@dms(u.N,k) ).toarray()

# def chebyshev_stiff(u=None,k=2,**kwargs):
#     '''  <Ti'' Tj> '''
#     return chebyshev_grad(u,k)

# def chebdirichlet_mass(u=None,**kwargs):
#     ''' 
#     Eq. (2.5) of Shen - Effcient Spectral-Galerkin Method II.
#     '''
#     diag0 = [1.5, *[1.0]*(u.N-4), 1.5]
#     diag2 = [*[-0.5]*(u.N-4) ]
#     return diags([diag2, diag0, diag2], [-2, 0, 2]).toarray()

# def chebdirichlet_stiff(u=None,**kwargs):
#     ''' 
#     Eq. (2.6) of Shen - Effcient Spectral-Galerkin Method II.
    
#     Equivalent to
#         S@D2@Si
#         S, Si : Stencil matrix and its inverse
#         D2:     Stiffness matrix of the Chebyshev Basis <T''T> 
#     '''
#     N,M = u.N,u.N
#     D = np.zeros( (N,M) )
#     for m in range(N):
#         for n in range(m,M):
#             if (n==m):
#                 D[m,n] = -2*(m+1)*(m+2)
#             elif (n-m)%2==0:
#                 D[m,n] = -4*(m+1)
#     return to_sparse(D).toarray()

# _list_of_known_inner_products ={
#     "CH^0,CH^0": chebyshev_mass,
#     "CH^0,CH^1": chebyshev_grad,
#     "CH^0,CH^2": chebyshev_stiff,

#     "CD^0,CD^0": chebdirichlet_mass,
#     "CD^0,CD^2": chebdirichlet_stiff,
# }



# def size_of_gen(gen):
#     ''' Determine size of iterator. 
#     Better to clone gen before it enters this function.'''
#     try:
#         return len(gen)
#     except TypeError:
#         return sum(1 for _ in gen)



# def inner(u,v,w="GL"):
#     '''
#     Inner product of <u,v>_w, where w is the weight.
    
#     The function can handle 3 scenarios:
#     1) If u&v are arrays, the result is a scalar.
#     2) if u is a class that implements .get_basis and 
#     v is an array the result is an array
#     3) If u&v implement .get_basis, the result is a
#     mass matrix
    
    
#     Input
#         u: ndarray or method that implements .get_basis()
#         v: ndarray or method that implements .get_basis()
#         w: ndarray or "GL" (Gauss-Lobatto weight)
#     Output
#         scalar, array or matrix depending on input
#     '''
#     # -- check inputs
#     if all([isinstance(i,np.ndarray) for i in [u,v]]):
#         method = "scalar"
#         assert u.size==v.size
#         N = u.size
#     elif all([hasattr(i,"get_basis") for i in [u,v]]):
#         method = "matrix"
#         assert u.N==v.N
#         N = u.N
        
#     elif hasattr(u,"get_basis") and isinstance(v,np.ndarray):
#         method = "array"
#         assert u.N==v.size
#         N = u.N
#     else:
#         raise NotImplementedError("Try t0 switch u and v.")
    
#     # -- handle weights
#     if w is None: 
#         w = np.ones(N)
#     elif isinstance(w,str):
#         if w == ("GL"):
#             w = np.concatenate(([0.5],np.ones(N-2),[0.5] ))
#         else:
#             raise NotImplementedError("Unknown weight key supplied.")
    
#     # -- inner product
#     if method=="scalar":
#         return np.sum(u*v*w)/(N-1)
    
#     if method=="array":
#         return np.array([inner(uu,v,w) 
#                 for uu in u.iter_basis()])
    
#     if method=="matrix":
#         M = []
#         for j,vv in enumerate(v.iter_basis()):
#             M.append( 
#                 [inner(uu,vv,w) 
#                 for uu in u.iter_basis()]
#                 )
#         return np.vstack(M)