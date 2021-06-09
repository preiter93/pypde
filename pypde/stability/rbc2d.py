from .decorator import *
from .utils import *
from .convolution import convolution_matrix2d as conv_mat
from ..field import Field
from ..bases.spectralbase import Base
import matplotlib.pyplot as plt
import numpy as np

# from scipy.linalg import eig
from scipy.sparse.linalg import eigs


def plot_evec(evecs, U, V, P, T, xx, yy, m=-1):
    u, v, p, t = split_evec(evecs, m=m)

    # U
    u = np.real(u).reshape(U.vhat.shape)
    u = U.backward(u)
    # V
    v = np.real(v).reshape(V.vhat.shape)
    v = V.backward(v)
    # T
    t = np.real(t).reshape(T.vhat.shape)
    t = T.backward(t)

    levels = np.linspace(t.min(), t.max(), 50)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, t, cmap="RdBu_r", levels=levels)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Quiver
    speed = np.max(np.sqrt(u ** 2 + v ** 2))
    # if skip is None:
    skip = u.shape[0] // 16
    ax.quiver(
        xx[::skip, ::skip],
        yy[::skip, ::skip],
        u[::skip, ::skip] / speed,
        v[::skip, ::skip] / speed,
        scale=7.9,
        width=0.007,
        alpha=0.5,
        headwidth=4,
    )

    plt.show()


# --------------------------------------------------------------------
@io_decorator
def solve_rbc2d(
    Nx=21,
    Ny=21,
    Ra=2600,
    Pr=1,
    aspect=1,
    sidewall="adiabatic",
    plot=True,
    norm_diff=True,
):
    """
    Solves the linear stability problem of convection-diffusion equation
    in a 2-D rectangular domain.

    Note:
    Works very good with odd Nx and Ny

    >>>
    import numpy as np
    from pypde import *
    from scipy.linalg import eig
    import matplotlib.pyplot as plt
    from pypde.stability.rbc2d import solve_rbc2d

    # Parameters
    Nx,Ny  = 21,21
    Ra     = 2500
    Pr     = 1.0
    aspect = 2.0

    # Find the growth rates for given Ra
    evals,evecs = solve_rbc2d(Nx=Nx,Ny=Ny,Ra=Ra,Pr=Pr,aspect=aspect)
    >>>
    """

    # ----------------------- Parameters ---------------------------
    shape = (Nx, Ny)

    U = Field([Base(shape[0], "CD"), Base(shape[1], "CD")])
    V = Field([Base(shape[0], "CD"), Base(shape[1], "CD")])
    T = Field([Base(shape[0], "CN"), Base(shape[1], "CD")])
    P = Field([Base(shape[0], "CN"), Base(shape[1], "CN")])
    CH = Field([Base(shape[0], "CH", dealias=2), Base(shape[1], "CH", dealias=2)])

    # Rescale by factor 2 (chebyshev)
    scale_z = 0.5
    x, y = U.x * scale_z * aspect, U.y * scale_z
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # -- Mean Field
    UU = np.zeros(CH.shape)
    VV = np.zeros(CH.shape)
    TT = np.zeros(CH.shape)

    TT[:] = CH.forward(-1.0 * yy)

    evals, evecs = solve_stability_2d(
        Ra,
        Pr,
        U,
        V,
        T,
        P,
        CH,
        UU,
        VV,
        TT,
        scale=(scale_z * aspect, scale_z),
        norm_diff=norm_diff,
    )

    if plot:
        plot_evec(evecs, U, V, P, T, xx, yy, m=-1)

    return evals, evecs


def solve_stability_2d(
    Ra, Pr, U, V, T, P, CH, UU, VV, TT, scale=(1, 1), norm_diff=True
):

    if norm_diff:
        # Normalization 0 (Diffusion)
        nu = Pr
        ka = 1
        gr = Pr * Ra
    else:
        # Normalization 1 (Turnover)
        nu = np.sqrt(Pr / Ra)
        ka = np.sqrt(1 / Pr / Ra)
        gr = 1.0

    # -- Matrices
    Ix, Iy = np.eye(CH.vhat.shape[0]), np.eye(CH.vhat.shape[1])
    D1x = CH.xs[0].dms(1) / scale[0]
    D1y = CH.xs[1].dms(1) / scale[1]
    D2x = CH.xs[0].dms(2) / scale[0] ** 2.0
    D2y = CH.xs[1].dms(2) / scale[1] ** 2.0

    # Derivative (2D)
    N = U.vhat.shape[0] * U.vhat.shape[1]
    I = np.eye(N)
    Dx1 = np.kron(D1x, Iy)
    Dx2 = np.kron(D2x, Iy)
    Dy1 = np.kron(Ix, D1y)
    Dy2 = np.kron(Ix, D2y)

    # Transforms (2D)
    SU = np.kron(U.xs[0].S, U.xs[1].S)
    SV = np.kron(V.xs[0].S, V.xs[1].S)
    ST = np.kron(T.xs[0].S, T.xs[1].S)
    SP = np.kron(P.xs[0].S, P.xs[1].S)
    SUT = np.kron(U.xs[0].ST, U.xs[1].ST)
    SVT = np.kron(V.xs[0].ST, V.xs[1].ST)
    STT = np.kron(T.xs[0].ST, T.xs[1].ST)
    SPT = np.kron(P.xs[0].ST, P.xs[1].ST)

    # Derivatives
    UU_x, UU_y = D1x @ UU, UU @ D1y.T
    VV_x, VV_y = D1x @ VV, VV @ D1y.T
    TT_x, TT_y = D1x @ TT, TT @ D1y.T

    # -- Build ----------------------

    # -- Diffusion + Non-Linear 2: Udu
    Udx = conv_mat(UU, field=CH) @ Dx1
    Vdy = conv_mat(VV, field=CH) @ Dy1

    L2d = -nu * (Dx2 + Dy2) + Udx + Vdy
    K2d = -ka * (Dx2 + Dy2) + Udx + Vdy

    L2d_u = SUT @ L2d @ SU
    L2d_v = SVT @ L2d @ SV
    K2d_t = STT @ K2d @ ST

    # -- Buoyancy Uz
    that = SVT @ ST

    # -- Pressure
    dpdx = SUT @ Dx1 @ SP
    dpdy = SVT @ Dy1 @ SP

    # -- Divergence
    dudx = SPT @ Dx1 @ SU
    dvdy = SPT @ Dy1 @ SV

    # -- Non-Linear 1: udU

    # dTdx
    dTdx = conv_mat(TT_x, field=CH)
    dTdx = STT @ dTdx @ SU

    # dTdy
    dTdy = conv_mat(TT_y, field=CH)
    dTdy = STT @ dTdy @ SV

    # dUdx
    dUdx = conv_mat(UU_x, field=CH)
    dUdx = SUT @ dUdx @ SU

    # dUdy
    dUdy = conv_mat(UU_y, field=CH)
    dUdy = SUT @ dUdy @ SV

    # dVdx
    dVdx = conv_mat(VV_x, field=CH)
    dVdx = SVT @ dVdx @ SU

    # dVdy
    dVdy = conv_mat(VV_y, field=CH)
    dVdy = SVT @ dVdy @ SV

    # -- Mass
    MU = SUT @ SU
    MV = SVT @ SV
    MT = STT @ ST

    # ------------
    # LHS
    L11 = L2d_u + dUdx
    L12 = dUdy
    L13 = dpdx
    L14 = 0 * I
    L21 = dVdx
    L22 = L2d_v + dVdy
    L23 = dpdy
    L24 = -gr * that
    L31 = dudx
    L32 = dvdy
    L33 = 0.0 * I
    L34 = 0.0 * I
    L41 = dTdx
    L42 = dTdy
    L43 = 0.0 * I
    L44 = K2d_t

    # RHS
    M11 = 1 * MU
    M12 = 0 * I
    M13 = 0 * I
    M14 = 0 * I
    M21 = 0 * I
    M22 = 1 * MV
    M23 = 0 * I
    M24 = 0 * I
    M31 = 0 * I
    M32 = 0 * I
    M33 = 0 * I
    M34 = 0 * I
    M41 = 0 * I
    M42 = 0 * I
    M43 = 0 * I
    M44 = 1 * MT

    L1 = np.block([[L11, L12, L13, L14]])
    M1 = np.block([[M11, M12, M13, M14]])  # u
    L2 = np.block([[L21, L22, L23, L24]])
    M2 = np.block([[M21, M22, M23, M24]])  # v
    L3 = np.block([[L31, L32, L33, L34]])
    M3 = np.block([[M31, M32, M33, M34]])  # p
    L4 = np.block([[L41, L42, L43, L44]])
    M4 = np.block([[M41, M42, M43, M44]])  # T

    L = np.block([[L1], [L2], [L3], [L4]])
    M = np.block([[M1], [M2], [M3], [M4]])
    L += np.eye(L.shape[0]) * 1e-20  # Make non-singular

    # -- Solve EVP ----
    # evals,evecs = eig(L,1.j*M)
    evals, evecs = eigs(L, k=8, M=M, sigma=0)  # shift and invert
    evals = evals * -1.0j

    # Post Process egenvalues
    evals, evecs = remove_evals(evals, evecs, higher=1400)
    evals, evecs = sort_evals(evals, evecs, which="I")

    return evals, evecs
