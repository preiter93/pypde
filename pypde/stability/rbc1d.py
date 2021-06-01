from .decorator import *
from .utils import *
from .convolution import convolution_matrix1d as conv_mat
from ..field import Field
from ..bases.spectralbase import Base
import matplotlib.pyplot as plt
import numpy as np 
from scipy.linalg import eig


#-------------------------------------------------------------------
@io_decorator
def solve_rbc1d(Ny=100,Ra=1708,Pr=1,alpha=3.14,plot=True,norm_diff=True):
    #----------------------- Parameters ---------------------------
    if norm_diff:
        # Normalization 0 (Diffusion)
        nu = Pr
        ka = 1
        beta = Pr*Ra
    else:
        # Normalization 1 (Turnover)
        nu = np.sqrt(Pr/Ra)
        ka = np.sqrt(1/Pr/Ra)
        beta = 1.0

    Lz = 2.0 # Size of Chebyshev Domain

    # -- Fields
    shape=(Ny,)
    U = Field( [Base(shape[0],"CD")] )
    V = Field( [Base(shape[0],"CD")] )
    T = Field( [Base(shape[0],"CD")] )
    P = Field( [Base(shape[0],"CN")] )
    CH = Field([Base(shape[0],"CH",dealias=2)] )
    z = U.x/Lz

    # -- Matrices 
    I = np.eye(U.vhat.shape[0])

    Dz = CH.xs[0].dms(1)*(Lz)
    Dx = 1.j*alpha*np.eye(CH.vhat.shape[0])
    Dz2 = Dz@Dz
    Dx2 = Dx@Dx


    # -- Mean Field
    UU = np.zeros(CH.shape)
    #VV = np.zeros(CH.shape)
    TT = np.zeros(CH.shape)

    TT[:] = CH.forward(-1.0*z) 
    #UU[:] = CH.forward(-1.0*z)

    # -- Build

    # -- Diffusion
    D2U = nu*U.xs[0].ST@(-Dx2 - Dz2)@U.xs[0].S
    D2V = nu*V.xs[0].ST@(-Dx2 - Dz2)@V.xs[0].S
    D2T = ka*T.xs[0].ST@(-Dx2 - Dz2)@T.xs[0].S

    # -- Buoyancy Uz
    BVT = V.xs[0].ST@T.xs[0].S

    # -- Pressure
    DXP = U.xs[0].ST@Dx@P.xs[0].S
    DZP = V.xs[0].ST@Dz@P.xs[0].S

    # -- Divergence
    DXU = P.xs[0].ST@Dx@U.xs[0].S
    DZV = P.xs[0].ST@Dz@V.xs[0].S

    # -- Non-Linear udU

    # dTdz
    dTdz = conv_mat(Dz@TT,field=CH)
    NTV = T.xs[0].ST@dTdz@V.xs[0].S

    # dUdz
    dUdz = conv_mat(Dz@UU,field=CH)
    NUV  = U.xs[0].ST@dUdz@V.xs[0].S

    # -- Non-Linear Udu
    UU = conv_mat(UU,field=CH)
    nUU = U.xs[0].ST@(UU@Dx)@U.xs[0].S
    nUV = V.xs[0].ST@(UU@Dx)@V.xs[0].S

    # -- Mass Matrices
    MU = U.xs[0].ST@U.xs[0].S
    MV = V.xs[0].ST@V.xs[0].S
    #MP = P.xs[0].ST@P.xs[0].S
    MT = T.xs[0].ST@T.xs[0].S

    # ------------
    # LHS
    L11 = 1.*D2U+1.*nUU  ; L12 = 1.*NUV         ; L13 = 1.*DXP   ; L14 = 0*I
    L21 = 0.*I           ; L22 = 1.*D2V+1.*nUV  ; L23 = 1.*DZP   ; L24 =-1.*beta*BVT
    L31 = 1.*DXU         ; L32 = 1.*DZV         ; L33 = 0.*I     ; L34 = 0.*I
    L41 = 0.*I           ; L42 = 1.*NTV         ; L43 = 0.*I     ; L44 = 1.*D2T

    # RHS
    M11 = 1*MU    ; M12 = 0*I     ; M13 = 0*I      ; M14 = 0*I
    M21 = 0*I     ; M22 = 1*MV    ; M23 = 0*I      ; M24 = 0*I
    M31 = 0*I     ; M32 = 0*I     ; M33 = 0*I      ; M34 = 0*I
    M41 = 0*I     ; M42 = 0*I     ; M43 = 0*I      ; M44 = 1*MT

    L1 = np.block([ [L11,L12,L13,L14] ]);  M1 = np.block([ [M11,M12,M13,M14] ]) #u
    L2 = np.block([ [L21,L22,L23,L24] ]);  M2 = np.block([ [M21,M22,M23,M24] ]) #v
    L3 = np.block([ [L31,L32,L33,L34] ]);  M3 = np.block([ [M31,M32,M33,M34] ]) #p
    L4 = np.block([ [L41,L42,L43,L44] ]);  M4 = np.block([ [M41,M42,M43,M44] ]) #T
        
    # -- Solve EVP ----
    L = np.block([ [L1], [L2], [L3], [L4]])
    M = np.block([ [M1], [M2], [M3], [M4]])
    evals,evecs = eig(L,1.j*M)

    # Post Process egenvalues
    evals, evecs = remove_evals(evals,evecs,higher=1400)
    evals, evecs = sort_evals(evals,evecs,which="I")

    if plot:
        blue = (0/255, 137/255, 204/255)
        red  = (196/255, 0, 96/255)
        yel   = (230/255,159/255,0)

        u,v,p,t = split_evec(evecs,m=-1)
        U.vhat[:] = np.real(u)
        V.vhat[:] = np.real(v)
        P.vhat[:] = np.real(p)
        T.vhat[:] = np.real(t)
        U.backward()
        V.backward()
        P.backward()
        T.backward()

        fig,(ax0,ax1,ax2) = plt.subplots(ncols=3, figsize=(8,3))
        ax0.set_title("Eigenvalues")
        ax0.set_xlim(-1,1);  ax0.grid(True)
        ax0.scatter(np.real(evals[:]),np.imag(evals[:]), marker="o", edgecolors="k", s=60, facecolors='none'); 

        ax1.set_ylabel("y"); ax1.set_title("Largest Eigenvector")
        ax1.plot(U.v,z,  marker="", color=blue, label=r"$|u|$")
        ax1.plot(V.v,z,  marker="", color=red,  label=r"$|v|$")
        #ax2.plot(P.v,z,  marker="", color="k" , label=r"$|p|$")
        ax1.legend(loc="lower right")
        ax2.set_ylabel("y"); ax2.set_title("Largest Eigenvector")
        ax2.plot(T.v,z,  marker="", color=yel , label=r"$|T|$")
        ax2.legend()
        plt.tight_layout(); 

    return evals,evecs