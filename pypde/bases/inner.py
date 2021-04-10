import numpy as np
from types import GeneratorType
from itertools import tee


def inner(u,v,N=None,w="GL"):
    '''
    Inner product of <u,v>_w, where w is the weight.
    
    The function can handle 3 scenarios:
    1)  u: array
        v: array
    2)  u: generator (over basis or derivative of basis)
        v: array
    3)  u: generator
        v: generator
    
    
    Input
        u: ndarray or method that implements .get_basis()
        v: ndarray or method that implements .get_basis()
        N: Size of u and v
        w: ndarray or "GL" (Gauss-Lobatto weight)
    Output
        (1) scalar, (2) array or (3) matrix
    '''
    # -- check inputs
    if all([isinstance(i,np.ndarray) for i in [u,v]]):
        method = "scalar"
        assert u.size==v.size
        if N is None: N = u.size
    elif all([isinstance(i,GeneratorType) for i in [u,v]]):
        method = "matrix"
        if N is None:
            u,u_clone=tee(u)
            N = size_of_gen(u_clone)
        #N = 10
    elif isinstance(u,GeneratorType) and isinstance(v,np.ndarray):
        method = "array"
        if N is None: N = v.size
    elif isinstance(v,GeneratorType) and isinstance(u,np.ndarray):
        return inner(v,u,N,w) # Switch u and v
    else:
        raise NotImplementedError("Try t0 switch u and v.")

    # -- handle weights
    if w is None: 
        w = np.ones(N)
    elif isinstance(w,str):
        if w == ("GL"):
            w = np.concatenate(([0.5],np.ones(N-2),[0.5] ))
        else:
            raise NotImplementedError("Unknown weight key supplied.")
    
    # -- inner product
    if method=="scalar":
        return np.sum(u*v*w)/(N-1)
    
    if method=="array":
        return np.array([inner(uu,v,N,w) for uu in u])
    
    if method=="matrix":
        M = []
        for j,vv in enumerate(v):
            u,u_clone = tee(u) # clone iterator
            c = [inner(uu,vv,N,w) for uu in u]
            M.append(c)
            u = u_clone

        return np.vstack(M)

def size_of_gen(gen):
    ''' Determine size of iterator. 
    Better to clone gen before it enters this function.'''
    try:
        return len(gen)
    except TypeError:
        return sum(1 for _ in gen)



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