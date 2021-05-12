import numpy as np

def TDMA(a,b,c,d):
    '''
    Tridiagonal matrix solver to solve
        Ax = d
    A is banded with diagonals in offsets -1, 0, 1

    Input
        a,b,c: 1d arrays
            diagonals -1, 0, 1
        d: nd array
            rhs (solved along axis 0)

    Return
        array
    '''
    n = len(d)
    w = np.zeros(n-1,float)
    g = np.zeros(d.shape,float)
    p = np.zeros(d.shape,float)

     # Forward sweep
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]
    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])

    # Back substitution
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p

def TDMA_offset(a,b,c,d,k):
    '''
    Tridiagonal matrix solver to solve
            Ax = d
    where A is banded with diagonals in offsets -k, 0, k

    Input
        a,b,c: 1d arrays
            diagonals -k, 0, k
        d: nd array
            rhs (solved along axis 0)
        k: int
            Offset of sub-diagonal

    Return
        array

    Test
    > from pypde.bases.linalg.tdma import TDMA_offset as TDMA
    > N = 10
    > l = np.random.rand(N-2)
    > d = np.random.rand(N)
    > u = np.random.rand(N-2)
    > b = np.random.rand(N)
    > A = np.diag(l,-2) + np.diag(d,0) + np.diag(u,2)
    > x = TDMA(l,d,u,b,2)
    > assert np.allclose(x,np.linalg.solve(A,b))
    '''
    n = len(d)
    w = np.zeros(n-2,float)
    g = np.zeros(d.shape, float)
    p = np.zeros(d.shape,float)

    # Forward sweep
    for i in range(n-k):
        if i<k:
            w[i] = c[i]/b[i]
        else:
            w[i] = c[i]/(b[i] - a[i-k]*w[i-k])

    for i in range(n):
        if i<k:
            g[i] = d[i]/b[i]
        else:
            g[i] = (d[i] - a[i-k]*g[i-k])/(b[i] - a[i-k]*w[i-k])

    # Back substitution
    p[n-k-1:n] = g[n-k-1:n]
    for i in range(n-k,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i+k-1]
    return p


def TDMA_Fortran(a,b,c,d,k):
    from .fortran import tdma
    if d.ndim==1:
        return tdma.solve_tdma_1d(a,b,c,d,int(k))
    elif d.ndim==2:
        return tdma.solve_tdma_2d(a,b,c,d,int(k))
    else:
        raise ValueError("TDMA Fortran supports only ndim<3 at the moment!")
