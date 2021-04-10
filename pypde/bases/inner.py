import numpy as np

def inner(u,v,w="GL"):
    '''
    Inner product of <u,v>_w, where w is the weight.
    
    The function can handle 3 scenarios:
    1) If u&v are arrays, the result is a scalar.
    2) if u is a class that implements .get_basis and 
    v is an array the result is an array
    3) If u&v implement .get_basis, the result is a
    mass matrix
    
    
    Input
        u: ndarray or method that implements .get_basis()
        v: ndarray or method that implements .get_basis()
        w: ndarray or "GL" (Gauss-Lobatto weight)
    Output
        scalar, array or matrix depending on input
    '''
    # -- check inputs
    if all([isinstance(i,np.ndarray) for i in [u,v]]):
        method = "scalar"
        assert u.size==v.size
        N = u.size
    elif all([hasattr(i,"get_basis") for i in [u,v]]):
        method = "matrix"
        assert u.N==v.N
        N = u.N
        
    elif hasattr(u,"get_basis") and isinstance(v,np.ndarray):
        method = "array"
        assert u.N==v.size
        N = u.N
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
        return np.array([inner(uu,v,w) 
                for uu in u.iter_basis()])
    
    if method=="matrix":
        M = []
        for j,vv in enumerate(v.iter_basis()):
            M.append( 
                [inner(uu,vv,w) 
                for uu in u.iter_basis()]
                )
        return np.vstack(M)