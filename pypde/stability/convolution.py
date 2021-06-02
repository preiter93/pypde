import numpy as np
from ..field import Field


def convolution(a, b, field):
    """
    Calculate Convolution

    Input
        a,b: nd array
            Spectral Coefficients
        field: Field
            Field, for spectral Transform
    """
    A = field.dealias.backward(a)
    B = field.dealias.backward(b)

    return field.dealias.forward(A * B)


def convolution_matrix1d(a, field):
    """
    Construct Convolution matrix by passing multiple
    delta functions to the convolution function

    >>>
    shape = (20,)
    B = Field([Base(shape[0],"CH",dealias=2)] )
    a = np.sin(B.x)+1
    b = np.cos(B.x)+1
    c = a*b
    ahat = B.forward(a)
    bhat = B.forward(b)
    M = convolution_matrix1d(ahat,field=B)
    chat = M@bhat
    np.allclose(B.forward(c), chat,1e-4)
    >>>
    """
    N = a.shape[0]
    b = np.zeros(N)
    M = np.zeros((N, N))
    for i in range(N):
        b[:] = 0.0
        b[i] = 1.0
        M[i, :] = convolution(a, b, field)
    return M.T


def convolution_matrix2d(a, field):
    """
    Construct Convolution matrix by passing multiple
    delta functions to the convolution function

    >>>
    shape = (4,5)
    B = Field([Base(shape[0],"CH",dealias=2),Base(shape[1],"CH",dealias=2)] )
    xx,yy = np.meshgrid(B.x,B.y,indexing="ij")
    a = np.sin(xx)+1
    b = np.cos(yy)+1
    c = a*b
    ahat = B.forward(a)
    bhat = B.forward(b)
    chat = B.forward(c)

    M = convolution_matrix2d(ahat,field=B)
    chat = M@bhat.reshape(-1)
    np.allclose(B.forward(c), chat.reshape(ahat.shape))
    >>>
    """
    Nx, Ny = a.shape
    b = np.zeros((Nx, Ny))
    M = np.zeros((Nx * Ny, Nx * Ny))
    for i in range(Nx):
        for j in range(Ny):
            b[:] = 0.0
            b[i, j] = 1.0
            M[j + (i * Ny), :] = convolution(a, b, field).flatten()
    return M.T


# shape = (20,)
# B = Field([Base(shape[0],"CH",dealias=2)] )
# a = np.sin(B.x)+1
# b = np.cos(B.x)+1
# c = a*b
# ahat = B.forward(a)
# bhat = B.forward(b)
# M = convolution_matrix(ahat,field=B)
# chat = M@bhat
# np.allclose(B.forward(c), chat,1e-4)
