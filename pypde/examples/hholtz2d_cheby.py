import numpy as np
import matplotlib.pyplot as plt
from pypde.solver.matrix import *
from pypde.solver.operator import *
from pypde.field import SpectralField

LAM = 1/np.pi**6 

def _f(xx,yy,lam=LAM):
    return  (1.0+2*lam*np.pi**2/4)*_u(xx,yy)

def _u(xx,yy):
    return np.cos(np.pi/2*xx)*np.cos(np.pi/2*yy)


class SolverHHoltz2dChebyshev():
    '''
    Solve Helmholtz equation in 2-dimension using 
    chebyshev polynomials.

            (1-lam*nabla^2) u = f

    In 2 dimensions, the alternating direction implicit (ADI)
    is used to separate x & y direction.

    Input:
        field: SpectralField
            Field variable u 
        rhs: SpectralField
            Force field variable f 
        lam: float 
        	Prefactor of nabla^2
    '''
    def __init__(self,field,rhs,lam):
        assert field.dim == 2, "2-Dimensional Solver!"
        assert np.all([isinstance(i,SpectralField) for i in 
            [field,rhs]]), "SolveHHoltz needs SpectralField for initialization."
        self.shape = field.shape_spectral

        # --- Matrices ----
        Sx = field.xs[0].S
        Bx = field.xs[0].B(2)@Sx
        Ix = field.xs[0].I()@Sx
        Ax =  Bx-lam*Ix

        Sy = field.xs[1].S
        By = field.xs[1].B(2)@Sy
        Iy = field.xs[1].I()@Sy
        Ay =  By-lam*Iy

        # --- RHS ---------
        self.b = RHSExplicit()
        self.b.add_PM(MatrixRHS(Bx,axis=0))
        self.b.add_PM(MatrixRHS(By,axis=1))

        # --- LHS ----------
        Ax = MatrixLHS(A=Ax,ndim=2,axis=0, solver="fdma")
        Ay = MatrixLHS(A=Ay,ndim=2,axis=1, solver="fdma")
        self.A = LHSImplicit(Ax)
        self.A.add(Ay)

    def solve(self,rhs):
        assert rhs.shape == self.shape
        self.b.b = rhs 
        return self.A.solve(self.b.rhs)

N = 40
shape = (N+2,N+2)

# ----------- Dirichlet -----------------
u = SpectralField(shape, ("CD","CD"))
f = SpectralField(shape, ("CD","CD"))
x,y = u.x,u.y; xx,yy = np.meshgrid(x,y)

HHoltz = SolverHHoltz2dChebyshev(u,f,lam=LAM)
f.v = _f(xx,yy)
f.forward()
fhat = f.vhat

u.vhat = HHoltz.solve(fhat)
u.backward()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx,yy,u.v, rstride=1, cstride=1, cmap="viridis",edgecolor="k")
plt.show()