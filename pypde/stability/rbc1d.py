from .decorator import *
from .utils import *
from .convolution import convolution_matrix1d as conv_mat
from ..field import Field
from ..bases.spectralbase import Base
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig


# -------------------------------------------------------------------
@io_decorator
def solve_rbc1d(Ny=41, Ra=1708, Pr=1, alpha=3.14, plot=True, norm_diff=True):
    """
    Solves the linear stability problem of convection-diffusion equation
    in a 2-D rectangular domain.

    Note:
    Ny should be odd

    >>>
    import numpy as np
    from pypde import *
    from scipy.linalg import eig
    import matplotlib.pyplot as plt
    from pypde.stability.rbc1d import solve_rbc1d
    # Parameters
    Ny    = 41
    alpha = 3.14
    Ra    = 1710
    Pr    = 1.0

    # Find the growth rates for given Ra
    evals,evecs = solve_rbc1d(Ny=Ny,Ra=Ra,Pr=Pr,alpha=alpha,plot=True)
    >>>
    """
    # ----------------------- Parameters ---------------------------
    # Correct for size of Chebyshev Domain, which is 2, not 1
    # Ra /= 2**3
    # alpha /= 2

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

    # -- Fields
    shape = (Ny,)
    U = Field([Base(shape[0], "CD")])
    V = Field([Base(shape[0], "CD")])
    T = Field([Base(shape[0], "CD")])
    P = Field([Base(shape[0], "CN")])
    CH = Field([Base(shape[0], "CH", dealias=2)])
    # Rescale by factor 2 (chebyshev)
    scale_z = 2.0
    y = U.x / scale_z

    # -- Matrices
    I = np.eye(U.vhat.shape[0])

    Dy = CH.xs[0].dms(1) * scale_z
    Dx = 1.0j * alpha * np.eye(CH.vhat.shape[0])
    Dy2 = Dy @ Dy
    Dx2 = Dx @ Dx

    # -- Mean Field
    UU = np.zeros(CH.shape)
    # VV = np.zeros(CH.shape)
    TT = np.zeros(CH.shape)

    TT[:] = CH.forward(-1.0 * y)

    # -- Build

    # -- Diffusion + Non-Linear 2: Udu
    Udx = conv_mat(UU) @ Dx

    L2d = -nu * (Dx2 + Dy2) + Udx
    K2d = -ka * (Dx2 + Dy2) + Udx
    L2d_u = U.xs[0].ST @ L2d @ U.xs[0].S
    L2d_v = V.xs[0].ST @ L2d @ V.xs[0].S
    K2d_t = T.xs[0].ST @ K2d @ T.xs[0].S

    # -- Buoyancy Uz
    that = V.xs[0].ST @ T.xs[0].S

    # -- Pressure
    dpdx = U.xs[0].ST @ Dx @ P.xs[0].S
    dpdy = V.xs[0].ST @ Dy @ P.xs[0].S

    # -- Divergence
    dudx = P.xs[0].ST @ Dx @ U.xs[0].S
    dvdy = P.xs[0].ST @ Dy @ V.xs[0].S

    # -- Non-Linear 1: udU

    # dTdy
    # dTdy = -1.0*np.eye(CH.vhat.shape[0])
    dTdy = conv_mat(Dy @ TT)
    dTdy = T.xs[0].ST @ dTdy @ V.xs[0].S

    # dUdy
    dUdy = conv_mat(Dy @ UU)
    dUdy = U.xs[0].ST @ dUdy @ V.xs[0].S
    dUdy = 0.0 * I

    # -- Mass Matrices
    MU = U.xs[0].ST @ U.xs[0].S
    MV = V.xs[0].ST @ V.xs[0].S
    MT = T.xs[0].ST @ T.xs[0].S

    # ------------
    # LHS
    L11 = L2d_u
    L12 = dUdy
    L13 = dpdx
    L14 = 0 * I
    L21 = 0.0 * I
    L22 = L2d_v
    L23 = dpdy
    L24 = -1.0 * gr * that
    L31 = dudx
    L32 = dvdy
    L33 = 0.0 * I
    L34 = 0.0 * I
    L41 = 0.0 * I
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

    # -- Solve EVP ----
    L = np.block([[L1], [L2], [L3], [L4]])
    M = np.block([[M1], [M2], [M3], [M4]])
    L += np.eye(L.shape[0]) * 1e-20  # Make non-singular
    evals, evecs = eig(L, 1.0j * M)

    # Post Process egenvalues
    evals, evecs = remove_evals(evals, evecs, higher=1400)
    evals, evecs = sort_evals(evals, evecs, which="I")

    if plot:
        blue = (0 / 255, 137 / 255, 204 / 255)
        red = (196 / 255, 0, 96 / 255)
        yel = (230 / 255, 159 / 255, 0)

        u, v, p, t = split_evec(evecs, m=-1)
        U.vhat[:] = np.real(u)
        V.vhat[:] = np.real(v)
        P.vhat[:] = np.real(p)
        T.vhat[:] = np.real(t)
        U.backward()
        V.backward()
        P.backward()
        T.backward()

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(11, 3))
        ax0.set_title("Eigenvalues")
        ax0.set_xlim(-1, 1)
        ax0.grid(True)
        ax0.scatter(
            np.real(evals[:]),
            np.imag(evals[:]),
            marker="o",
            edgecolors="k",
            s=60,
            facecolors="none",
        )

        ax1.set_ylabel("y")
        ax1.set_title("Largest Eigenvector")
        ax1.plot(U.v, y, marker="", color=blue, label=r"$|u|$")
        ax1.plot(V.v, y, marker="", color=red, label=r"$|v|$")
        ax1.legend(loc="lower right")
        ax2.set_ylabel("y")
        ax2.set_title("Largest Eigenvector")
        ax2.plot(T.v, y, marker="", color=yel, label=r"$|T|$")
        ax2.legend()
        ax3.set_ylabel("y")
        ax2.set_title("Largest Eigenvector")
        ax3.plot(P.v, y, marker="", color="k", label=r"$|P|$")
        ax3.legend()
        plt.tight_layout()

    return evals, evecs
