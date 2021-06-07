from pypde import *
import numpy as np
import matplotlib.pyplot as plt


# class Integrator():

#     def update(self):
#         raise NotImplementedError

#     def iterate(self,maxtime):
#         ''' Iterate till max time'''
#         while self.time<maxtime:

#             self.update()
#             self.update_time()

#             if ( (self.time+1e-3*self.dt)%self.tsave<self.dt*0.5):
#                 self.save()
#                 print("Time: {:5.3f}".format(self.time))

#                 # if self.field.check():
#                 #     print("\nNan or large value detected! STOP\n")
#                 #     break


# def update_time(self):
#     self.field.t += self.dt
#     self.time += self.dt

# def save(self):
#     self.field.save()


class Diffusion1d(Integrator):
    CONFIG = {
        "N": 50,
        "kappa": 1.0,
        "tsave": 0.01,
        "dt": 0.2,
        "ndim": 1,
        "base": "CD",
    }

    def __init__(self, **kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        self.time = 0.0
        # Field
        xbase = Base(self.N, self.base)
        self.field = Field([xbase])
        # Solver
        # self.setup_solver()
        self.solver_from_template()

    def setup_solver(self):
        """The same as solverplan_hholtz1d below"""
        # --- Matrices ----
        Sx = self.field.xs[0].S_sp
        Bx = self.field.xs[0].family.B(2, 2)
        Ix = self.field.xs[0].family.I(2)
        lam = self.dt * self.kappa
        Ax = Bx @ Sx - lam * Ix @ Sx

        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_rhs(PlanRHS(Bx, ndim=1, axis=0))  # rhs
        solver.add_old(PlanRHS(Bx @ Sx, ndim=1, axis=0))  # rhs old velocity
        solver.add_lhs(PlanLHS(Ax, ndim=1, axis=0, method="fdma"))  # lhs
        solver.show_plan()

        self.solver = solver
        self.Sx = Sx

    def solver_from_template(self):
        from pypde.templates.hholtz import solverplan_hholtz1d

        self.solver = solverplan_hholtz1d(self.field.xs, lam=self.dt * self.kappa)

    @property
    def _f(self):
        """Forcing"""
        return np.cos(np.pi / 2 * self.field.x)

    @property
    @memoized
    def _fhat(self):
        return self.field.xs[0].family.forward_fft(self._f)
        # return self.field.xs[0].forward_fft(self._f)

    def update(self):
        # Solve
        rhs = self.solver.solve_rhs(self.dt * self._fhat)
        rhs += self.solver.solve_old(self.field.vhat)
        self.field.vhat[:] = self.solver.solve_lhs(rhs)


D = Diffusion1d(N=50, dt=0.001, tsave=0.1)
D.iterate(5)

anim = D.field.animate(D.field.x, duration=4)
plt.show()
