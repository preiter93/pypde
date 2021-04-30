import numpy as np

def TDMA(a,b,c,d):
    ''' 
    Tridiagonal matrix solver to solve
        Ax = d
    A is banded with diagonals in offsets -1, 0, 1

    Input
        a,b,c: array
            diagonals -1, 0, 1
        d: array
            solution vector (rhs)

    Return
        array
    '''
    n = len(d)
    w = np.zeros(n-1,float)
    g = np.zeros(n, float)
    p = np.zeros(n,float)
    
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
        a,b,c: array
            diagonals -k, 0, k
        d: array
            solution vector (rhs)

    Return
        array
    '''
    n = len(d)
    w = np.zeros(n-2,float)
    g = np.zeros(n, float)
    p = np.zeros(n,float)
    
    # Forward sweep
    w[0:k] = c[0:k]/b[0:k]
    g[0:k] = d[0:k]/b[0:k]
    for i in range(k,n-k):
        w[i] = c[i]/(b[i] - a[i-k]*w[i-k])

    for i in range(k,n):
        g[i] = (d[i] - a[i-k]*g[i-k])/(b[i] - a[i-k]*w[i-k])

    # Back substitution
    p[n-k-1:n] = g[n-k-1:n]
    for i in range(n-2,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i+k-1]
    return p
