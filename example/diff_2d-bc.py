import sys
sys.path.append("./")
from pypde import *
import numpy as np
import matplotlib.pyplot as plt


class Diffusion2d(Integrator):
    CONFIG={
        "bases": ("CD","CN"),
        "shape": (20,20),
        "kappa": 1.0,
        "tsave": 0.01,
        "dt": 0.2,
        "ndim": 2,
        "beta": 0.5,
    }
    def __init__(self,**kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        self.time = 0.0
        # Field
        xbase = Base(self.shape[0],self.bases[0])
        ybase = Base(self.shape[1],self.bases[1])
        self.field = Field([xbase,ybase])
        # Boundary Conditions
        self.setup_fieldbc()
        # Solver
        #self.setup_solver()
        self.solver_from_template()

        self.init_field()
        self.field.save()

        self.rhs = np.zeros(self.shape)

    def init_field(self):
        self.field.v[:] = 0#-self.fieldbc.v
        self.field.forward()
        self.field.backward()

    def setup_solver(self):
        lam = self.dt*self.kappa*self.beta
        # --- Matrices ----
        Sx = self.field.xs[0].S_sp
        Bx = self.field.xs[0].family.B(2,2)
        Ix = self.field.xs[0].family.I(2)
        Ax =  Bx@Sx-lam*Ix@Sx
        self.Sx = Sx

        Sy = self.field.xs[1].S_sp
        By = self.field.xs[1].family.B(2,2)
        Iy = self.field.xs[1].family.I(2)
        Ay =  By@Sy-lam*Iy@Sy
        self.Sy = Sy

        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_rhs(PlanRHS(Bx,ndim=2,axis=0))
        solver.add_rhs(PlanRHS(By,ndim=2,axis=1))
        solver.add_old(PlanRHS(Bx@Sx,ndim=2,axis=0))
        solver.add_old(PlanRHS(By@Sy,ndim=2,axis=1))
        solver.add_lhs(PlanLHS(Ax,ndim=2,axis=0,method="fdma") ) #lhs
        solver.add_lhs(PlanLHS(Ay,ndim=2,axis=1,method="fdma") ) #lhs
        solver.show_plan()

        self.solver = solver

    def solver_from_template(self):
        from pypde.templates.hholtz import solverplan_hholtz2d_adi
        self.solver = solverplan_hholtz2d_adi(self.field.xs,
            lam=self.dt*self.kappa*self.beta)

    def setup_fieldbc(self):
        ''' Setup Inhomogeneous field'''
        bc = np.zeros((2,self.shape[1])) # boundary condition
        bc[0,:] = np.cos(np.pi*self.field.y)
        fieldbc = FieldBC(self.field.xs,axis=0)
        fieldbc.add_bc(bc)
        self.fieldbc = fieldbc

    @property
    @memoized
    def _fhat(self):
        ''' rhs from inhomogeneous bcs'''
        fieldbc_d2 = grad(self.fieldbc,deriv=(0,2),return_field=True)
        return self.dt*self.kappa*fieldbc_d2.vhat#*self.beta

    def update(self):
        # Solve
        self.rhs[:] = self._fhat
        # Add diffusive term
        gradT = grad(self.field,deriv=(0,2),return_field=True)
        self.rhs += self.dt*self.kappa*(1.0-self.beta)*gradT.vhat
        gradT = grad(self.field,deriv=(2,0),return_field=True)
        self.rhs += self.dt*self.kappa*(1.0-self.beta)*gradT.vhat

        rhs  = self.solver.solve_rhs(self.rhs)
        rhs += self.solver.solve_old(self.field.vhat)
        self.field.vhat[:] = self.solver.solve_lhs(rhs)


D = Diffusion2d(shape=(50,50),dt=0.01,tsave=0.1,kappa=0.1,beta=0.5)
D.iterate(10.0)

# from mpl_toolkits import mplot3d
#
# xx,yy = np.meshgrid(D.field.x,D.field.y,indexing="ij")
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(xx, yy, D.field.V[-1], rstride=1, cstride=1,cmap="viridis")
# plt.show()
#
# gradT = grad(D.field,deriv=(0,2),return_field=True)
# gradT.backward()
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(xx, yy, gradT.v, rstride=1, cstride=1,cmap="viridis")
# plt.show()
#
# gradT = grad(D.field,deriv=(2,0),return_field=True)
# gradT.backward()
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(xx, yy, gradT.v, rstride=1, cstride=1,cmap="viridis")
# plt.show()

#  Add inhomogeneous part
for i,v in enumerate(D.field.V):
   D.field.V[i] += D.fieldbc.v

# plt.plot(D.field.x, D.field.V[-1][:,20])
# plt.show()

anim = D.field.animate(D.field.x,duration=4,wireframe=True)
plt.show()
