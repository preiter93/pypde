import numpy as np
from ..field import Field


def P_shift(N, m):
    """
    Literatur:
        Baszenski:
            Fast Polynomial multiplication and convolutions
            related to the discrete cosine transform
            (p15)
    """
    P = np.zeros((N, N))
    if m == 0:
        for i in range(0, N):
            P[i, i] = 1
        return P
    elif m == N - 1:
        for i in range(0, N):
            P[i, -i - 1] = 1
        return P

    P[0, m] = 1
    P[N - 1, N - m - 1] = 1
    for i in range(1, N - 1):
        if i < m + 1:
            P[i, m - i] = 0.5
        else:
            P[i, i - m] = 0.5

        if i < N - m:
            P[i, i + m] = 0.5
        else:
            P[i, 2 * (N - 1) - i - m] = 0.5
    return P


def circ(a):
    """
    Express convolution by matrix multiplication, such that
        a x b = circ(a) @ b
    where x stands for convolution.

    Literatur:
        Baszenski:
            Fast Polynomial multiplication and convolutions
            related to the discrete cosine transform
            (p15)
    """
    N = a.shape[0]

    a = a.copy()
    a[0] *= 2
    a[N - 1] *= 2

    M = np.zeros((N, N))
    c = np.ones(N)
    c[0] = c[-1] = 0.5

    j = 0
    for j in range(0, N):
        P = P_shift(N, j)
        M[j, :] = c[j] * P @ a
    return M


def circ2d(a):
    """
    Express convolution by matrix multiplication, such that
        a x b = circ(a) @ b
    where x stands for convolution.

    Literatur:
        Baszenski:
            Fast Polynomial multiplication and convolutions
            related to the discrete cosine transform
            (p15)
    """
    Nx, Ny = a.shape[0], a.shape[1]

    a2 = a.copy()
    a2[0, :] *= 2
    a2[-1, :] *= 2
    a2[:, 0] *= 2
    a2[:, -1] *= 2

    cx, cy = np.ones(Nx), np.ones(Ny)
    cx[0] = cx[-1] = cy[0] = cy[-1] = 0.5

    M = np.zeros((Nx * Ny, Nx * Ny))

    jx, jy = 0, 1
    for jx in range(0, Nx):
        for jy in range(0, Ny):
            Px = P_shift(Nx, jx)
            Py = P_shift(Ny, jy)
            # row = (Nx-1)*jx + jy
            row = jy + (jx * Ny)
            M[row, :] = (cx[jx] * cy[jy] * Px @ a2 @ Py.T).flatten()
    return M


def convolution_matrix1d(a):
    """
    Construct Convolution matrix for discrete cosine
    transform. See literature reference in circ()

    >>>
    from pypde import *

    shape = (5,)
    B = Field([Base(shape[0], "CH", dealias=2)])
    ahat = np.random.random(shape)
    bhat = np.random.random(shape)
    a = B.backward(ahat)
    b = B.backward(bhat)
    c = a * b
    ahat = B.forward(a)
    bhat = B.forward(b)
    M = convolution_matrix1d(ahat)
    chat = M @ bhat
    chat1 = B.forward(c)

    assert np.allclose(chat1, chat,1e-4)
    >>>
    """
    return circ(a)


def convolution_matrix2d(a):
    """
    Construct Convolution matrix for discrete cosine
    transform. See literature reference in circ2d()

    >>>
    from pypde import *

    shape = (4,3)
    B = Field([Base(shape[0],"CH",dealias=2),Base(shape[1],"CH",dealias=2)] )

    # Construct test matrices
    ahat = np.random.random(shape)
    bhat = np.random.random(shape)
    a = B.backward(ahat)
    b = B.backward(bhat)
    c = a*b
    ahat = B.forward(a)
    bhat = B.forward(b)
    chat = B.forward(c) # True solution

    # Test convolution routine
    M = convolution_matrix2d(ahat)
    chat_conv = M@bhat.flatten()
    assert np.allclose(chat.flatten(), chat_conv)
    >>>
    """

    return circ2d(a)


#
# # ###### OLD ROUTINES -
# def convolution(a, b, field):
#     """
#     Calculate Convolution
#
#     Input
#         a,b: nd array
#             Spectral Coefficients
#         field: Field
#             Field, for spectral Transform
#     """
#     A = field.dealias.backward(a)
#     B = field.dealias.backward(b)
#
#     return field.dealias.forward(A * B)
#
#
# def convolution_matrix1d(a, field):
#     """
#     Construct Convolution matrix by passing multiple
#     delta functions to the convolution function
#
#     >>>
#     shape = (20,)
#     B = Field([Base(shape[0],"CH",dealias=2)] )
#     a = np.sin(B.x)+1
#     b = np.cos(B.x)+1
#     c = a*b
#     ahat = B.forward(a)
#     bhat = B.forward(b)
#     M = convolution_matrix1d(ahat,field=B)
#     chat = M@bhat
#     np.allclose(B.forward(c), chat,1e-4)
#     >>>
#     """
#     N = a.shape[0]
#     b = np.zeros(N)
#     M = np.zeros((N, N))
#     for i in range(N):
#         b[:] = 0.0
#         b[i] = 1.0
#         M[i, :] = convolution(a, b, field)
#     return M.T
#
#
# def convolution_matrix2d(a, field):
#     """
#     Construct Convolution matrix by passing multiple
#     delta functions to the convolution function
#
#     >>>
#     shape = (4,5)
#     B = Field([Base(shape[0],"CH",dealias=2),Base(shape[1],"CH",dealias=2)] )
#     xx,yy = np.meshgrid(B.x,B.y,indexing="ij")
#     a = np.sin(xx)+1
#     b = np.cos(yy)+1
#     c = a*b
#     ahat = B.forward(a)
#     bhat = B.forward(b)
#     chat = B.forward(c)
#
#     M = convolution_matrix2d(ahat,field=B)
#     chat = M@bhat.reshape(-1)
#     np.allclose(B.forward(c), chat.reshape(ahat.shape))
#     >>>
#     """
#     Nx, Ny = a.shape
#     b = np.zeros((Nx, Ny))
#     M = np.zeros((Nx * Ny, Nx * Ny))
#     for i in range(Nx):
#         for j in range(Ny):
#             b[:] = 0.0
#             b[i, j] = 1.0
#             M[j + (i * Ny), :] = convolution(a, b, field).flatten()
#     return M.T
#

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
