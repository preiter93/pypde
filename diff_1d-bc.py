from pypde import *
import numpy as np
import matplotlib.pyplot as plt


class Diffusion1d(Integrator):
    CONFIG={
        "N": 50,
        "kappa": 1.0,
        "tsave": 0.01,
        "dt": 0.2,
        "ndim": 1,
    }
    def __init__(self,**kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        self.time = 0.0
        # Field
        shape = (self.N)
        self.field = Field(shape,("CD"))
        # Boundary Conditions
        self.setup_fieldbc()
        # Solver
        self.setup_solver_fast()

        self.init_field()
        self.field.save()

    def init_field(self):
        self.field.v = -self.fieldbc.v
        self.field.forward()
        self.field.backward()

    def setup_solver_fast(self):
        # --- Matrices ----
        Sx = self.field.xs[0].S_sp
        Bx = self.field.xs[0].family.B(2,2)
        Ix = self.field.xs[0].family.I(2)
        lam = self.dt*self.kappa
        Ax =  Bx@Sx-lam*Ix@Sx
        
        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_old(PlanRHS(Bx@Sx,ndim=1,axis=0)) # rhs old velocity
        solver.add_lhs(PlanLHS(Ax,ndim=1,axis=0,method="fdma") ) #lhs
        solver.show_plan()

        self.solver = solver

    def setup_solver(self):
        # --- Matrices ----
        Dx = self.field.xs[0].stiff
        Mx = self.field.xs[0].mass
        lam = self.dt*self.kappa
        Ax =  Mx - lam*Dx

        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_old(PlanRHS(Mx,ndim=1,axis=0)) # rhs old velocity
        solver.add_lhs(PlanLHS(Ax,ndim=1,axis=0,method="numpy") ) #lhs
        solver.show_plan()

        self.solver = solver

    
    def setup_fieldbc(self):
        ''' Setup Inhomogeneous field'''
        # boundary conditions
        bc = np.zeros(2)
        bc[1] = 1
        shape,bases = (self.N),("CD")
        fieldbc = FieldBC(shape,bases,axis=0)
        fieldbc.add_bc(bc)
        self.fieldbc = fieldbc


    @property
    @memoized
    def _fhat(self):
        ''' rhs from inhomogeneous bcs'''
        fieldbc_d2 = derivative_field(self.fieldbc,deriv=(2))
        return self.dt*self.kappa*fieldbc_d2.vhat

    def update(self):
        # Solve
        #rhs = self.solver.solve_rhs(self._fhat)
        rhs = self.solver.solve_old(self.field.vhat)
        self.field.vhat[:] = self.solver.solve_lhs(rhs)


D = Diffusion1d(N=250,dt=0.01,tsave=0.1,kappa=0.1)
D.iterate(5.0)

#  Add inhomogeneous part
for i,v in enumerate(D.field.V):
   D.field.V[i] += D.fieldbc.v

anim = D.field.animate(D.field.x,duration=4)
plt.show()