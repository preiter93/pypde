import numpy as np
import matplotlib.pyplot as plt
from pypde.solver.matrix import *
from pypde.solver.operator import *
from pypde.field import SpectralField
from pypde.utils import eigen_decomp

ARG = np.pi/2

def _f(xx,yy):
        return np.sin(ARG*xx)*np.sin(ARG*yy)
def _u(xx,yy):
        return np.sin(ARG*xx)*np.sin(ARG*yy)*-1/ARG**2/2


class SolverPoisson2dChebyshev():
    '''
    Solve Poisson equation in 2-dimension using 
    chebyshev polynomials.

            nabla^2 u = f

    Input:
        field: SpectralField
            Field variable u 
        rhs: SpectralField
            Force field variable f 
        pure_neumann: bool (optional)
            Handle singularity of pure_neumann poisson
    '''
    def __init__(self,field,rhs,pure_neumann=False,):
        assert field.dim == 2, "2-Dimensional Solver!"
        assert np.all([isinstance(i,SpectralField) for i in 
            [field,rhs]]), "SolvePoisson needs SpectralField for initialization."
        self.shape = field.shape_spectral

        # --- Matrices ----
        Sx = field.xs[0].S
        Bx = field.xs[0].B(2)@Sx
        Ax = field.xs[0].I()@Sx

        Sy = field.xs[1].S
        By = field.xs[1].B(2)@Sy
        Ay = field.xs[1].I()@Sy

        ByI = np.linalg.inv(By)
        wy,Qy,Qyi = eigen_decomp(Ay.T@ByI.T)

        # --- RHS ---------
        self.b = RHSExplicit()
        self.b.add_PM(MatrixRHS(Bx,axis=0))
        self.b.add_PM(MatrixRHS(Qy.T,axis=1)) 

        # --- LHS ----------
        Ax = MatrixLHS(A=Bx,ndim=2,axis=0,
            solver="poisson",lam=wy,C=Ax,
            pure_neumann=pure_neumann)
        Ay = MatrixLHS(A=Qyi.T,ndim=2,axis=1,
            solver="matmul")
        self.A = LHSImplicit(Ax)
        self.A.add(Ay)

    def solve(self,rhs):
        assert rhs.shape == self.shape
        self.b.b = rhs 
        return self.A.solve(self.b.rhs)
        
N = 40
shape = (N+2,N+2)

# ----------- Neumann -----------------
u = SpectralField(shape, ("CN","CN"))
f = SpectralField(shape, ("CN","CN"))
x,y = u.x,u.y; xx,yy = np.meshgrid(x,y)

Poisson = SolverPoisson2dChebyshev(u,f,pure_neumann=True)
f.v = _f(xx,yy)
f.forward()
fhat = f.vhat

u.vhat = Poisson.solve(fhat)
u.backward()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx,yy,u.v, rstride=1, cstride=1, cmap="viridis",edgecolor="k")
plt.show()


# ----------- Dirichlet -----------------
u = SpectralField(shape, ("CD","CD"))
f = SpectralField(shape, ("CD","CD"))
x,y = u.x,u.y; xx,yy = np.meshgrid(x,y)

Poisson = SolverPoisson2dChebyshev(u,f,pure_neumann=False)
f.v = _f(xx,yy)
f.forward()
fhat = f.vhat

u.vhat = Poisson.solve(fhat)
u.backward()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx,yy,u.v, rstride=1, cstride=1, cmap="viridis",edgecolor="k")
plt.show()

# ----------- Mixed -----------------
u = SpectralField(shape, ("CD","CN"))
f = SpectralField(shape, ("CD","CN"))
x,y = u.x,u.y; xx,yy = np.meshgrid(x,y)

Poisson = SolverPoisson2dChebyshev(u,f,pure_neumann=False)
f.v = _f(xx,yy)
f.forward()
fhat = f.vhat

u.vhat = Poisson.solve(fhat)
u.backward()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx,yy,u.v, rstride=1, cstride=1, cmap="viridis",edgecolor="k")
plt.show()