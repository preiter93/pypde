import numpy as np
from scipy.linalg import toeplitz

# -- Full dmsuite for python: https://github.com/labrosse/dmsuite

def chebdif(ncheb, mder,L=2):
    """
    Calculate differentiation matrices using Chebyshev collocation.
    Returns the differentiation matrices D1, D2, .. Dmder corresponding to the
    mder-th derivative of the function f, at the ncheb Chebyshev nodes in the
    interval [-1,1].
    Parameters
    ----------
    ncheb : int, polynomial order. ncheb + 1 collocation points
    mder   : int
          maximum order of the derivative, 0 < mder <= ncheb - 1
    Returns
    -------
    x  : ndarray
         (ncheb + 1) x 1 array of Chebyshev points
    DM : ndarray
         mder x ncheb x ncheb  array of differentiation matrices
    Notes
    -----
    This function returns  mder differentiation matrices corresponding to the
    1st, 2nd, ... mder-th derivates on a Chebyshev grid of ncheb points. The
    matrices are constructed by differentiating ncheb-th order Chebyshev
    interpolants.
    The mder-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    .. math::
    f^{(m)}_i = D^{(m)}_{ij}f_j
    The code implements two strategies for enhanced accuracy suggested by
    W. Don and S. Solomonoff :
    (a) the use of trigonometric  identities to avoid the computation of
    differences x(k)-x(j)
    (b) the use of the "flipping trick"  which is necessary since sin t can
    be computed to high relative precision when t is small whereas sin (pi-t)
    cannot.
    It may, in fact, be slightly better not to implement the strategies
    (a) and (b). Please consult [3] for details.
    This function is based on code by Nikola Mirkov
    http://code.google.com/p/another-chebpy
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487
    Examples
    --------
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix.
    """
    stretch=2.0/L 
    ncheb-=1

    if mder >= ncheb + 1:
        raise Exception('number of nodes must be greater than mder')

    if mder <= 0:
        raise Exception('derivative order must be at least 1')

    DM = np.zeros((mder, ncheb + 1, ncheb + 1))
    # indices used for flipping trick
    nn1 = int(np.floor((ncheb + 1) / 2))
    nn2 = int(np.ceil((ncheb + 1) / 2))
    k = np.arange(ncheb+1)
    # compute theta vector
    th = k * np.pi / ncheb

    # Compute the Chebyshev points

    # obvious way
    #x = np.cos(np.pi*np.linspace(ncheb-1,0,ncheb)/(ncheb-1))
    # W&R way
    x = np.sin(np.pi*(ncheb - 2 * np.linspace(ncheb, 0, ncheb + 1))/(2 * ncheb))
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(th/2, (ncheb + 1, 1))
    # trigonometric identity
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)
    # flipping trick
    DX[nn1:, :] = -np.flipud(np.fliplr(DX[0:nn2, :]))
    # diagonals of D
    DX[range(ncheb + 1), range(ncheb + 1)] = 1.
    DX = DX.T

    # matrix with entries c(k)/c(j)
    C = toeplitz((-1.)**k)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    # Z contains entries 1/(x(k)-x(j))
    Z = 1 / DX
    # with zeros on the diagonal.
    Z[range(ncheb + 1), range(ncheb + 1)] = 0.

    # initialize differentiation matrices.
    D = np.eye(ncheb + 1)

    for ell in range(mder):
        # off-diagonals
        D = (ell + 1) * Z * (C * np.tile(np.diag(D), (ncheb + 1, 1)).T - D)
        # negative sum trick
        D[range(ncheb + 1), range(ncheb + 1)] = -np.sum(D, axis=1)
        # store current D in DM
        DM[ell, :, :] = D

    D = DM[mder-1,:,:]
    x, D = x[::-1], D[::-1,::-1] #x [1,-1] to x [-1,1]  for convenience)
    return x/stretch, D*stretch**mder


def fourdif(nfou, mder,L=2*np.pi):
    """
    Fourier spectral differentiation.
    Spectral differentiation matrix on a grid with nfou equispaced points in [0,2pi)
    INPUT
    -----
    nfou: Size of differentiation matrix.
    mder: Derivative required (non-negative integer)
    OUTPUT
    -------
    xxt: Equispaced points 0, 2pi/nfou, 4pi/nfou, ... , (nfou-1)2pi/nfou
    ddm: mder'th order differentiation matrix
    Explicit formulas are used to compute the matrices for m=1 and 2.
    A discrete Fouier approach is employed for m>2. The program
    computes the first column and first row and then uses the
    toeplitz command to create the matrix.
    For mder=1 and 2 the code implements a "flipping trick" to
    improve accuracy suggested by W. Don and A. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The flipping trick is necesary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.
    S.C. Reddy, J.A.C. Weideman 1998.  Corrected for MATLAB R13
    by JACW, April 2003.
    """
    stretch=2*np.pi/L 
    # grid points
    xxt = 2*np.pi*np.arange(nfou)/nfou
    # grid spacing
    dhh = 2*np.pi/nfou

    nn1 = int(np.floor((nfou-1)/2.))
    nn2 = int(np.ceil((nfou-1)/2.))

    # mder>2 actually works better with simple matrix multiplication instead of fft
    multi = 1
    if mder != 0:
        if mder%2 != 0:
            multi,der = mder, 1
        else:
            multi,der = mder//2, 2

    if der == 0:
        # compute first column of zeroth derivative matrix, which is identity
        col1 = np.zeros(nfou)
        col1[0] = 1
        row1 = np.copy(col1)

    elif der == 1:
        # compute first column of 1st derivative matrix
        col1 = 0.5*np.array([(-1)**k for k in range(1, nfou)], float)
        if nfou%2 == 0:
            topc = 1/np.tan(np.arange(1, nn2+1)*dhh/2)
            col1 = col1*np.hstack((topc, -np.flipud(topc[0:nn1])))
            col1 = np.hstack((0, col1))
        else:
            topc = 1/np.sin(np.arange(1, nn2+1)*dhh/2)
            col1 = np.hstack((0, col1*np.hstack((topc, np.flipud(topc[0:nn1])))))
        # first row
        row1 = -col1

    elif der == 2:
        # compute first column of 1st derivative matrix
        col1 = -0.5*np.array([(-1)**k for k in range(1, nfou)], float)
        if nfou%2 == 0:
            topc = 1/np.sin(np.arange(1, nn2+1)*dhh/2)**2.
            col1 = col1*np.hstack((topc, np.flipud(topc[0:nn1])))
            col1 = np.hstack((-np.pi**2/3/dhh**2-1/6, col1))
        else:
            topc = 1/np.tan(np.arange(1, nn2+1)*dhh/2)/np.sin(np.arange(1, nn2+1)*dhh/2)
            col1 = col1*np.hstack((topc, -np.flipud(topc[0:nn1])))
            col1 = np.hstack(([-np.pi**2/3/dhh**2+1/12], col1))
        # first row
        row1 = col1

    ddm = toeplitz(col1, row1)
    ddm = np.linalg.matrix_power(ddm, multi)
    return xxt/stretch, ddm*stretch**mder



# ------------------------------------------------
# Important Functions for Chebyshev Polynomials
# ------------------------------------------------


def gauss_lobatto(n):
    ''' Return Chebyshev-Gauss-Lobatto grid points'''
    k = np.linspace(n, 0, n + 1)
    return np.sin(np.pi*(n - 2 * k)/(2 * n))

#@memoized
def diff_mat_spectral(N,deriv):
    '''Derivative matrix in spectral space of classical Chebyshev 
    polynomial on Gauss Lobattor points, see
    Jan S. Hesthaven - Spectral Methods for Time-Dependent Problems (p. 256)  

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
    if deriv==3: 
        for n in range(N):
            for p in range(n+3,N):
                if (p+n)%2!=0:
                    p2,n2 = p**2,n**2
                    D[n,p] = p*(p2*(p2-2) - 2*p2*n2 +(n2-1)**2)
        D /= 4
    if deriv==4:
        for n in range(N):
            for p in range(n+4,N):
                if (p+n)%2==0:
                    p2,n2 = p**2,n**2
                    p4,n4 = p**4,n**4
                    D[n,p] = p*(
                        p2*(p2-4)**2 - 3*p4*n2 + 3*p2*n4-n2*(n2-4)**2)
        D /= 24
    if deriv>4:
        raise NotImplementedError("derivatives > 4 not implemented")
    D[0,:] *= 0.5
    return D

def diff_recursion_spectral(c,deriv):
    ''' Recursion formula for computing coefficients 
    of deriv'th derivative of classical Chebyshev polynomial
    on Gauss Lobatto points

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

# ------------------------------------------------
# Unused Routines
# ------------------------------------------------

def fourdifft(f,M,L=2*np.pi):
    """
    Differentiation of f defined on equispaced grid
    via fft based on chebdifft.m (matlab)
    INPUT
    -----
    f: Function (ndarray dim 1)
    M: Order of derivative required (non-negative integer)
    """
    N=len(f)
    stretch=2*np.pi/L 

    k=np.fft.fftfreq(N,d=(1.0 / N ))
    k=(k*complex(0,1)*stretch)**M 

    fhat = np.fft.fft(f)
    dfhat = fhat*k
    return np.real(  np.fft.ifft(dfhat) )



def chebdifft(f,M,L=2):
    """
    Differentiation of f defined on chebyshev grid 
    via fft based on chebdifft.m (matlab)
    INPUT
    -----
    f: Function (ndarray dim 1)
    M: Order of derivative required (non-negative integer)
    """
    N=len(f)
    stretch=2/L 
    
    a = np.flipud(f[1:N-1])
    a = np.concatenate((f,a))
    a0 = np.fft.fft(a*stretch**M)

    ones = np.ones(N-2)
    a = np.concatenate(([0.5],ones,[0.5] ))
    a0 = a0[0:N]*a/(N-1)  #a0 contains Chebyshev coefficients of f

    # Recursion formula for computing coefficients of ell'th derivative 
    a = np.zeros((N,M+1),dtype="complex_")
    a[:,0] = a0
    for ell in np.arange(1,M+1):
        a[N-ell-1,ell]=2*(N-ell)*a[N-ell,ell-1];
        for k in np.arange(N-ell-2,0,-1):
            a[k,ell]=a[k+2,ell]+2*(k+1)*a[k+1,ell-1]
        a[0,ell]=a[1,ell-1]+a[2,ell]/2

    # Transform back to nphysical space
    b1 = [2*a[0,M]]
    b2 = a[1:N-1,M]
    b3 = [2*a[N-1,M]]
    b4 = np.flipud(b2)
    back = np.concatenate((b1,b2,b3,b4))
    df = 0.5*np.fft.fft(back)
    # Real data in, real derivative out
    df = df[0:N]*(-1)**(M%2)
    return np.real(df)