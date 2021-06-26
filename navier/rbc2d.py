import sys

sys.path.append("./")
from pypde import *
from pypde.plot import *

try:
    from rbc2d_base import *
except:
    from .rbc2d_base import *
# import matplotlib.pyplot as plt
import time

initplot()

TIME = 0
TIME_U = 0
TIME_V = 0
TIME_P = 0
TIME_T = 0
TIME_Update = 0
TIME_Divergence = 0

TIME_FFT = 0
TIME_Conv = 0


class NavierStokes(
    NavierStokesBase, NavierStokesSteadyState, NavierStokesStability, Integrator
):
    """
    Solve Navier Stokes equation + temperature equation.

    rbc:
        Adiabatic Sidewall

    linear:
        Isothermal Sidewall + Linear distribution

    zero:
        Isothermal Sidewall + Zero sidewall temperature
    """

    avail_cases = ["rbc", "linear", "zero"]

    def __init__(self, case="rbc", **kwargs):
        if case not in self.avail_cases:
            raise ValueError("Specified case is not available: ", self.avail_cases)
        else:
            self.case = case
        NavierStokesBase.__init__(self, **kwargs)
        Integrator.__init__(self)

        if self.case == "rbc":
            side = "CN"
        else:
            side = "CD"

        if self.case == "zero":
            self.set_fieldbc = self.set_temp_fieldbc_zero
        else:
            self.set_fieldbc = self.set_temp_fieldbc_linear

        # Space for Fields
        self.T = Field(
            [
                Base(self.shape[0], side, dealias=3 / 2),
                Base(self.shape[1], "CD", dealias=3 / 2),
            ]
        )
        self.U = Field(
            [
                Base(self.shape[0], "CD", dealias=3 / 2),
                Base(self.shape[1], "CD", dealias=3 / 2),
            ]
        )
        self.V = Field(
            [
                Base(self.shape[0], "CD", dealias=3 / 2),
                Base(self.shape[1], "CD", dealias=3 / 2),
            ]
        )
        self.P = Field([Base(self.shape[0], "CN"), Base(self.shape[1], "CN")])

        # Additional pressure field
        self.pres = Field([Base(self.shape[0], "CH"), Base(self.shape[1], "CH")])

        # Store list of fields for collective saving and time update
        self.field = MultiField(
            [self.T, self.U, self.V, self.P], ["temp", "ux", "uz", "pres"]
        )

        # Setup Solver solverplans
        self.setup_solver()

        # Setup Temperature field and bcs
        # self.set_temperature()
        self.set_fieldbc()

        # Array for rhs
        self.rhs = np.zeros(self.shape)

        # Add stability solver
        NavierStokesSteadyState.__init__(self)
        NavierStokesStability.__init__(self)

    def reset(self, reset_time=True):
        """
        Reset nu, kappa and solvers.
        Call after Ra or Pr has changed
        """
        self.set_nu_kappa()
        self.setup_solver()
        if reset_time:
            self.time = 0.0
            for field in self.field.fields:
                field.time = 0.0

    def set_temperature(self, amplitude=0.5, m=1):
        self.T.v = amplitude * np.sin(m * np.pi * self.xx) * np.cos(np.pi * self.yy)
        self.T.forward()

    def set_velocity(self, amplitude=0.5, m=1, n=1):
        x = (self.x - self.x[0]) / (self.x[-1] - self.x[0])
        y = (self.y - self.y[0]) / (self.y[-1] - self.y[0])
        xx, yy = np.meshgrid(x, y, indexing="ij")
        self.U.v = -amplitude * np.sin(m * np.pi * xx) * np.cos(n * np.pi * yy)
        self.V.v = amplitude * np.cos(m * np.pi * xx) * np.sin(n * np.pi * yy)
        self.U.forward()
        self.V.forward()

    def set_temp_fieldbc_linear(self):
        """Setup Inhomogeneous field for temperature"""
        # Boundary Conditions T (y=-1; y=1)
        bc = np.zeros((self.shape[0], 2))
        bc[:, 0], bc[:, 1] = 0.5, -0.5
        self.Tbc = FieldBC(self.T.xs, axis=1)
        self.Tbc.add_bc(bc)

        # Second Derivative for Diffusion term
        self.dTbcdz2 = self.grad(self.Tbc, deriv=(0, 2))
        # First Derivative for Convective term
        vhat = self.grad(self.Tbc, deriv=(0, 1))
        if self.dealias:
            self.dTbcdz1 = self.deriv_field.dealias.backward(vhat)
        else:
            self.dTbcdz1 = self.deriv_field.backward(vhat)

        # Tbc in derivative space for Buoyancy
        self.Tbc_cheby = galerkin_to_cheby(self.Tbc.vhat, self.Tbc)

    def set_temp_fieldbc_zero(self):
        """Setup Inhomogeneous field for temperature"""
        # Boundary Conditions T (y=-1; y=1)
        bc = np.zeros((2, self.shape[1]))
        bc[0, :] = transfer_function(0.5, 0, -0.5, self.y, k=0.02)
        bc[1, :] = bc[0, :]

        # plt.plot(self.y, bc[0, :])
        # plt.show()

        self.Tbc = FieldBC(self.T.xs, axis=0)
        self.Tbc.add_bc(bc)

        # Second Derivative for Diffusion term
        self.dTbcdz2 = self.grad(self.Tbc, deriv=(0, 2))
        # First Derivative for Convective term
        vhat = self.grad(self.Tbc, deriv=(0, 1))
        if self.dealias:
            self.dTbcdz1 = self.deriv_field.dealias.backward(vhat)
        else:
            self.dTbcdz1 = self.deriv_field.backward(vhat)

        # Tbc in derivative space for Buoyancy
        self.Tbc_cheby = galerkin_to_cheby(self.Tbc.vhat, self.Tbc)

    def setup_solver(self):
        from pypde.templates.hholtz import solverplan_hholtz2d_adi
        from pypde.templates.poisson import solverplan_poisson2d

        if self.integrator == "rk3":
            # print("Initialize rk3 ...")
            self.set_timestep_coefficients_rk3()
        else:
            # print("Initialize euler ...")
            self.set_timestep_coefficients_euler()

        self.solver_U, self.solver_V, self.solver_T = [], [], []
        for rk in range(self.nstage):
            solver_U = solverplan_hholtz2d_adi(
                bases=self.U.xs,
                lam=self.dt * self.a[rk] * self.beta * self.nu,
                scale=self.scale,
            )
            solver_V = solverplan_hholtz2d_adi(
                bases=self.V.xs,
                lam=self.dt * self.a[rk] * self.beta * self.nu,
                scale=self.scale,
            )
            solver_T = solverplan_hholtz2d_adi(
                bases=self.T.xs,
                lam=self.dt * self.a[rk] * self.beta * self.kappa,
                scale=self.scale,
            )
            self.solver_U.append(solver_U)
            self.solver_V.append(solver_V)
            self.solver_T.append(solver_T)
        self.solver_P = solverplan_poisson2d(self.P.xs, singular=True, scale=self.scale)

    def update_velocity(self, p, u, v, fac=1.0):
        tic = time.perf_counter()

        dpdx = self.grad(p, deriv=(1, 0))
        dpdz = self.grad(p, deriv=(0, 1))

        u.vhat -= cheby_to_galerkin(dpdx * fac, u)
        v.vhat -= cheby_to_galerkin(dpdz * fac, v)

        global TIME_Update
        TIME_Update += time.perf_counter() - tic

    def divergence_velocity(self, u, v):
        tic = time.perf_counter()

        dudx = self.grad(u, deriv=(1, 0))
        dudz = self.grad(v, deriv=(0, 1))

        global TIME_Divergence
        TIME_Divergence += time.perf_counter() - tic

        return dudx + dudz

    def conv_term(self, field, ux, uz, add_bc=None):
        tic = time.perf_counter()
        conv = convective_term(
            field,
            ux,
            uz,
            deriv_field=self.deriv_field,
            add_bc=add_bc,
            dealias=self.dealias,
            scale=self.scale,
        )
        global TIME_Conv
        TIME_Conv += time.perf_counter() - tic

        return conv

    def update_U(self, stage):
        tic = time.perf_counter()

        # Pressure term
        dpdx = self.grad(self.pres, deriv=(1, 0))
        rhs = -self.dt * self.a[stage] * dpdx

        # Non-Linear Convection
        rhs -= self.dt * self.b[stage] * self.conv_term(self.U, self.ux, self.uz)
        if self.c[stage] != 0:
            rhs -= (
                self.dt
                * self.c[stage]
                * self.conv_term(self.U, self.ux_old, self.uz_old)
            )

        # Add explicit diffusive term
        if self.beta != 1.0:
            rhs += (
                self.dt
                * self.a[stage]
                * (1 - self.beta)
                * self.nu
                * self.grad(self.U, deriv=(2, 0))
            )
            rhs += (
                self.dt
                * self.a[stage]
                * (1 - self.beta)
                * self.nu
                * self.grad(self.U, deriv=(0, 2))
            )

        rhs = self.solver_U[stage].solve_rhs(rhs)
        rhs += self.solver_U[stage].solve_old(self.U.vhat)
        self.U.vhat[:] = self.solver_U[stage].solve_lhs(rhs)

        global TIME_U
        TIME_U += time.perf_counter() - tic

    def update_V(self, That, stage):
        tic = time.perf_counter()

        # Pressure term
        dpdz = self.grad(self.pres, deriv=(0, 1))
        rhs = -self.dt * self.a[stage] * dpdz

        # Non-Linear Convection
        rhs -= self.dt * self.b[stage] * self.conv_term(self.V, self.ux, self.uz)
        if self.c[stage] != 0:
            rhs -= (
                self.dt
                * self.c[stage]
                * self.conv_term(self.V, self.ux_old, self.uz_old)
            )

        # Buoyancy
        rhs += self.dt * self.a[stage] * That

        # Add explicit diffusive term
        if self.beta != 1.0:
            rhs += (
                self.dt
                * self.a[stage]
                * (1 - self.beta)
                * self.nu
                * self.grad(self.V, deriv=(2, 0))
            )
            rhs += (
                self.dt
                * self.a[stage]
                * (1 - self.beta)
                * self.nu
                * self.grad(self.V, deriv=(0, 2))
            )

        rhs = self.solver_V[stage].solve_rhs(rhs)
        rhs += self.solver_V[stage].solve_old(self.V.vhat)
        self.V.vhat[:] = self.solver_V[stage].solve_lhs(rhs)

        global TIME_V
        TIME_V += time.perf_counter() - tic

    def update_T(self, stage):
        tic = time.perf_counter()

        # Non-Linear Convection
        rhs = (
            -self.dt
            * self.b[stage]
            * self.conv_term(self.T, self.ux, self.uz, add_bc=self.uz * self.dTbcdz1)
        )
        if self.c[stage] != 0:
            rhs -= (
                self.dt
                * self.c[stage]
                * self.conv_term(
                    self.T, self.ux_old, self.uz_old, add_bc=self.uz_old * self.dTbcdz1
                )
            )

        # Add diffusion from bc's
        rhs += self.dt * self.a[stage] * self.kappa * self.dTbcdz2

        # Add explicit diffusive term
        if self.beta != 1.0:
            rhs += (
                self.dt
                * self.a[stage]
                * (1 - self.beta)
                * self.kappa
                * self.grad(self.T, deriv=(2, 0))
            )
            rhs += (
                self.dt
                * self.a[stage]
                * (1 - self.beta)
                * self.kappa
                * self.grad(self.T, deriv=(0, 2))
            )

        rhs = self.solver_T[stage].solve_rhs(rhs)
        rhs += self.solver_T[stage].solve_old(self.T.vhat)
        self.T.vhat[:] = self.solver_T[stage].solve_lhs(rhs)

        global TIME_T
        TIME_T += time.perf_counter() - tic

    def update_P(self, div, singular=True):
        tic = time.perf_counter()
        rhs = self.solver_P.solve_rhs(div)
        self.P.vhat[:] = self.solver_P.solve_lhs(rhs)
        if singular:
            self.P.vhat[0, 0] = 0

        global TIME_P
        TIME_P += time.perf_counter() - tic

    def update_pres(self, div, stage):
        self.pres.vhat -= 1.0 * self.nu * div * self.beta
        self.pres.vhat += (
            1.0 / (self.dt * self.a[stage]) * galerkin_to_cheby(self.P.vhat, self.P)
        )

    def update(self):
        self.ux_old, self.uz_old = 0, 0
        for rk in range(self.nstage):
            # Buoyancy
            That = galerkin_to_cheby(self.T.vhat, self.T)
            That += self.Tbc_cheby

            # Convection velocity
            tic = time.perf_counter()
            if self.dealias:
                self.ux = self.U.dealias.backward(self.U.vhat)
                self.uz = self.V.dealias.backward(self.V.vhat)
            else:
                self.ux = self.U.backward(self.U.vhat)
                self.uz = self.V.backward(self.V.vhat)

            global TIME_FFT
            TIME_FFT += time.perf_counter() - tic

            # Solve Ux
            self.update_U(stage=rk)
            self.update_V(That, stage=rk)

            # Divergence of Velocity
            div = self.divergence_velocity(self.U, self.V)

            # Solve Pressure
            self.update_P(div)

            # Update pressure
            self.update_pres(div, stage=rk)

            # Correct Velocity
            self.update_velocity(self.P, self.U, self.V)

            # Solve Temperature
            self.update_T(stage=rk)

            self.ux_old, self.uz_old = self.ux, self.uz


def transfer_function(TL, TM, TR, x, k=0.01):
    arr = np.zeros(x.shape)
    L = x[-1] - x[0]
    for i in range(x.size):
        xs = x[i] * 2.0 / L
        if xs < 0:
            arr[i] = -k * xs / (k + xs + 1) * (TL - TM) + TM
        else:
            arr[i] = k * xs / (k - xs + 1) * (TR - TM) + TM
    return arr


# class NavierStokesZero(NavierStokes):
#     """
#     Convection with zero Sidewall BC's
#     CONFIG={
#         "shape": (50,50),
#         "kappa": 1.0,
#         "nu": 1.0,
#         "dt": 0.2,
#         "ndim": 2,
#         "tsave": 0.1,
#         "dealias": True,
#         "integrator": "eu",
#         "beta": 1.0,

#     >>>
#     shape = (64,64)
#     Ra = 1e3
#     Pr = 1

#     NS = NavierStokesZero(
#             shape=shape,
#             dt=0.005,
#             tsave=1.0,
#             Ra=Ra,
#             Pr=Pr,
#             dealias=True,
#             integrator="rk3",
#             beta=0.5,
#             aspect=1.,
#         )
#     >>>
#     }
#     """

#     def __init__(self, adiabatic=False, **kwargs):
#         NavierStokes.__init__(self, adiabatic=adiabatic, **kwargs)

#     def set_temp_fieldbc(self):
#         """Setup Inhomogeneous field for temperature"""
#         # Boundary Conditions T (y=-1; y=1)
#         bc = np.zeros((2, self.shape[1]))
#         bc[0, :] = transfer_function(0.5, 0, -0.5, self.y, k=0.02)
#         bc[1, :] = bc[0, :]

#         # plt.plot(self.y, bc[0, :])
#         # plt.show()

#         self.Tbc = FieldBC(self.T.xs, axis=0)
#         self.Tbc.add_bc(bc)

#         # Second Derivative for Diffusion term
#         self.dTbcdz2 = self.grad(self.Tbc, deriv=(0, 2))
#         # First Derivative for Convective term
#         vhat = self.grad(self.Tbc, deriv=(0, 1))
#         if self.dealias:
#             self.dTbcdz1 = self.deriv_field.dealias.backward(vhat)
#         else:
#             self.dTbcdz1 = self.deriv_field.backward(vhat)

#         # Tbc in derivative space for Buoyancy
#         self.Tbc_cheby = galerkin_to_cheby(self.Tbc.vhat, self.Tbc)

#     def callback(self):
#         print(
#             "Divergence: {:4.2e}".format(
#                 np.linalg.norm(self.divergence_velocity(self.U, self.V))
#             )
#         )
#         self.eval_Nu()
