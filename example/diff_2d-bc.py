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
    }
    def __init__(self,**kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        self.time = 0.0
        # Field
        self.field = Field(self.shape,self.bases)
        # Boundary Conditions
        self.setup_fieldbc()
        # Solver
        self.setup_solver()

        self.init_field()
        self.field.save()

    def init_field(self):
        self.field.v = -self.fieldbc.v
        self.field.forward()
        self.field.backward()

    def setup_solver(self):
        lam = self.dt*self.kappa
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
    
    def setup_fieldbc(self):
        ''' Setup Inhomogeneous field'''
        bc = np.zeros((2,self.shape[1])) # boundary condition
        bc[0,:] = np.cos(np.pi*self.field.y)
        fieldbc = FieldBC(self.shape,self.bases,axis=0)
        fieldbc.add_bc(bc)
        self.fieldbc = fieldbc

    @property
    @memoized
    def _fhat(self):
        ''' rhs from inhomogeneous bcs'''
        fieldbc_d2 = derivative_field(self.fieldbc,deriv=(0,2))
        return self.dt*self.kappa*fieldbc_d2.vhat

    def update(self):
        # Solve
        rhs  = self.solver.solve_rhs(self._fhat)
        rhs += self.solver.solve_old(self.field.vhat)
        self.field.vhat[:] = self.solver.solve_lhs(rhs)


D = Diffusion2d(shape=(50,50),dt=0.01,tsave=0.1,kappa=0.1)
D.iterate(1.0)

#  Add inhomogeneous part
for i,v in enumerate(D.field.V):
   D.field.V[i] += D.fieldbc.v

anim = D.field.animate(D.field.x,duration=4)
plt.show()