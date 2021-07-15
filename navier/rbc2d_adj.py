import sys

sys.path.append("./")
from pypde import *
from pypde.plot import *

try:
    from rbc2d_base import NavierStokesBase
    from rbc2d import NavierStokes
except:
    from .rbc2d_base import NavierStokesBase
    from .rbc2d import NavierStokes


class NavierStokesAdjoint(NavierStokesBase, Integrator):
    """
    CONFIG={
        "shape": (50,50),
        "kappa": 1.0,
        "nu": 1.0,
        "dt": 0.2,
        "ndim": 2,
        "tsave": 0.1,
        "dealias": True,
        "integrator": "eu",
        "beta": 1.0,
        "Lx": 1.0,
    }
    """

    avail_cases = ["rbc", "linear", "zero"]

    def __init__(self, case="rbc", **kwargs):

        if case not in self.avail_cases:
            raise ValueError("Specified case is not available: ", self.avail_cases)
        else:
            self.case = case

        Integrator.__init__(self)
        NavierStokesBase.__init__(self, **kwargs)

        if self.case == "rbc":
            side = "CN"
        else:
            side = "CD"

        # Initialize underlying NavierStokes Solver
        self.NS = NavierStokes(case=case, **self.CONFIG)

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
            [self.T, self.U, self.V], ["temp", "ux", "uy"]
        )

        # Space for Adjoint Fields
        self.TA = Field(
            [
                Base(self.shape[0], side, dealias=3 / 2),
                Base(self.shape[1], "CD", dealias=3 / 2),
            ]
        )
        self.UA = Field(
            [
                Base(self.shape[0], "CD", dealias=3 / 2),
                Base(self.shape[1], "CD", dealias=3 / 2),
            ]
        )
        self.VA = Field(
            [
                Base(self.shape[0], "CD", dealias=3 / 2),
                Base(self.shape[1], "CD", dealias=3 / 2),
            ]
        )

        # Setup Solver solverplans
        self.setup_solver()

        # Setup Temperature field and bcs
        # self.set_temperature()
        self.Tbc = self.NS.Tbc
        self.Tbc.backward()

        # Array for rhs's
        self.rhs = np.zeros(self.shape)

        # Temperature BC in physical space
        self.deriv_field.vhat[:] = galerkin_to_cheby(self.NS.Tbc.vhat, self.NS.Tbc)
        if self.dealias:
            self.temp_bc = self.deriv_field.dealias.backward(self.deriv_field.vhat)
        else:
            self.temp_bc = self.deriv_field.backward(self.deriv_field.vhat)

    def reset_time(self):
        self.time = 0.0
        for field in self.field.fields:
            field.time = 0.0

    def set_temperature(self, amplitude=0.5):
        self.T.v = (
            amplitude * np.sin(0.5 * np.pi * self.xx) * np.cos(0.5 * np.pi * self.yy)
        )
        self.T.forward()

    def setup_solver(self):
        from pypde.templates.poisson import solverplan_poisson2d

        # Time step coefficients
        self.a = self.NS.a
        self.b = self.NS.b
        self.c = self.NS.c
        self.nstage = self.NS.nstage

        # Solver Plans
        self.solver_P = self.NS.solver_P

        # Plans to smooth fields
        self.nabla_U = solverplan_poisson2d(self.U.xs, singular=False, scale=self.scale)
        self.nabla_V = solverplan_poisson2d(self.V.xs, singular=False, scale=self.scale)
        self.nabla_T = solverplan_poisson2d(self.T.xs, singular=False, scale=self.scale)

    def save(self):
        self.T.save()
        self.P.save()
        self.U.save()
        self.V.save()

    def conv(self, field, u, deriv):
        return conv_term(
            field,
            u,
            deriv=deriv,
            deriv_field=self.deriv_field,
            dealias=self.dealias,
            scale=self.scale,
        )

    def conv_term_adj_ux(self, fieldx, fieldz, fieldT, ux, uz, temp, add_bc=None):

        conv = self.conv(fieldx, ux, deriv=(1, 0))

        conv += self.conv(fieldz, uz, deriv=(1, 0))

        conv += self.conv(fieldT, temp, deriv=(1, 0))

        conv += self.conv(fieldT, self.temp_bc, deriv=(1, 0))

        if self.dealias:
            return self.deriv_field.dealias.forward(conv)
        return self.deriv_field.forward(conv)

    def conv_term_adj_uz(self, fieldx, fieldz, fieldT, ux, uz, temp, add_bc=None):
        conv = self.conv(fieldx, ux, deriv=(0, 1))

        conv += self.conv(fieldz, uz, deriv=(0, 1))

        conv += self.conv(fieldT, temp, deriv=(0, 1))

        conv += self.conv(fieldT, self.temp_bc, deriv=(0, 1))

        if self.dealias:
            return self.deriv_field.dealias.forward(conv)
        return self.deriv_field.forward(conv)

    def update_NS(self):
        # -- Calculate Residual of Navier-Stokes
        self.NS.U.vhat[:] = self.U.vhat[:]
        self.NS.V.vhat[:] = self.V.vhat[:]
        self.NS.T.vhat[:] = self.T.vhat[:]
        self.NS.update()
        self.NS.U.vhat[:] = (self.NS.U.vhat[:] - self.U.vhat[:]) / self.dt
        self.NS.V.vhat[:] = (self.NS.V.vhat[:] - self.V.vhat[:]) / self.dt
        self.NS.T.vhat[:] = (self.NS.T.vhat[:] - self.T.vhat[:]) / self.dt

        # -- Smooth fields (nabla^2 TA = T)
        rhs = self.nabla_U.solve_rhs(galerkin_to_cheby(self.NS.U.vhat, self.NS.U))
        self.UA.vhat[:] = self.nabla_U.solve_lhs(rhs) / self.nu
        rhs = self.nabla_V.solve_rhs(galerkin_to_cheby(self.NS.V.vhat, self.NS.V))
        self.VA.vhat[:] = self.nabla_V.solve_lhs(rhs) / self.nu
        rhs = self.nabla_T.solve_rhs(galerkin_to_cheby(self.NS.T.vhat, self.NS.T))
        self.TA.vhat[:] = self.nabla_T.solve_lhs(rhs) / self.kappa

    def update_U(self, stage):
        self.rhs[:] = 0.0

        # Pressure term
        dpdx = self.grad(self.pres, deriv=(1, 0))
        self.rhs -= self.a[stage] * dpdx

        # Non-Linear Convection
        self.rhs += self.b[stage] * self.NS.conv_term(self.UA, self.ux, self.uz)
        self.rhs += self.b[stage] * self.conv_term_adj_ux(
            self.UA, self.VA, self.TA, self.ux, self.uz, self.temp
        )
        if self.c[stage] != 0:
            self.rhs += self.c[stage] * self.NS.conv_term(
                self.UA, self.ux_old, self.uz_old
            )
            self.rhs += self.c[stage] * self.conv_term_adj_ux(
                self.UA, self.VA, self.TA, self.ux_old, self.uz_old, self.temp_old
            )

        # Diffusion
        self.rhs += self.a[stage] * galerkin_to_cheby(self.NS.U.vhat, self.U)
        # self.rhs += self.a[stage] * self.nu * galerkin_to_cheby(self.NS.U.vhat, self.U)

        # Update
        self.U.vhat[:] += self.dt * cheby_to_galerkin(self.rhs, self.U)

    def update_V(self, stage):
        self.rhs[:] = 0.0

        # Pressure term
        dpdz = self.grad(self.pres, deriv=(0, 1))
        self.rhs -= self.a[stage] * dpdz

        # Non-Linear Convection
        self.rhs += self.b[stage] * self.NS.conv_term(self.VA, self.ux, self.uz)
        self.rhs += self.b[stage] * self.conv_term_adj_uz(
            self.UA, self.VA, self.TA, self.ux, self.uz, self.temp
        )
        if self.c[stage] != 0:
            self.rhs += self.c[stage] * self.NS.conv_term(
                self.VA, self.ux_old, self.uz_old
            )
            self.rhs += self.c[stage] * self.conv_term_adj_uz(
                self.UA, self.VA, self.TA, self.ux_old, self.uz_old, self.temp_old
            )

        # Diffusion
        # self.rhs += self.a[stage] * self.nu * galerkin_to_cheby(self.NS.V.vhat, self.V)
        self.rhs += self.a[stage] * galerkin_to_cheby(self.NS.V.vhat, self.V)

        # Update
        self.V.vhat[:] += self.dt * cheby_to_galerkin(self.rhs, self.V)

    def update_T(self, stage):
        self.rhs[:] = 0.0

        # Non-Linear Convection
        self.rhs += self.b[stage] * self.NS.conv_term(self.TA, self.ux, self.uz)
        if self.c[stage] != 0:
            self.rhs += self.c[stage] * self.NS.conv_term(
                self.TA, self.ux_old, self.uz_old
            )

        # Diffusion
        # self.rhs += (
        #    self.a[stage] * self.kappa * galerkin_to_cheby(self.NS.T.vhat, self.T)
        # )
        self.rhs += self.a[stage] * galerkin_to_cheby(self.NS.T.vhat, self.T)

        # Buoyancy
        self.rhs += self.a[stage] * galerkin_to_cheby(self.VA.vhat, self.VA)

        # Update
        self.T.vhat[:] += self.dt * cheby_to_galerkin(self.rhs, self.T)

    def update_P(self, div, singular=True):
        rhs = self.solver_P.solve_rhs(div)
        self.P.vhat[:] = self.solver_P.solve_lhs(rhs)
        if singular:
            self.P.vhat[0, 0] = 0

    def update_pres(self, div, stage):
        self.pres.vhat += galerkin_to_cheby(self.P.vhat, self.P) / (
            self.dt * self.a[stage]
        )

    def update_velocity(self, p, u, v, fac=1.0):

        dpdx = self.grad(p, deriv=(1, 0))
        dpdz = self.grad(p, deriv=(0, 1))

        u.vhat -= cheby_to_galerkin(dpdx * fac, u)
        v.vhat -= cheby_to_galerkin(dpdz * fac, v)

    def callback(self):
        self.eval_Nu()
        # -- Divergence
        print(
            "|div| = {:4.2e}".format(
                np.linalg.norm(self.NS.divergence_velocity(self.U, self.V))
            )
        )
        print(
            "|div residual|: {:4.2e}".format(
                np.linalg.norm(self.NS.divergence_velocity(self.NS.U, self.NS.V))
            )
        )

        # -- Norm
        print(" |U| = {:5.2e}".format(np.linalg.norm(self.NS.U.vhat)))
        print(" |V| = {:5.2e}".format(np.linalg.norm(self.NS.V.vhat)))
        print(" |T| = {:5.2e}".format(np.linalg.norm(self.NS.T.vhat)))

    def update(self):
        self.ux_old, self.uz_old, self.temp_old = 0, 0, 0
        for rk in range(self.nstage):

            # Convection velocity
            if self.dealias:
                self.ux = self.U.dealias.backward(self.U.vhat)
                self.uz = self.V.dealias.backward(self.V.vhat)
                self.temp = self.T.dealias.backward(self.T.vhat)
            else:
                self.ux = self.U.backward(self.U.vhat)
                self.uz = self.V.backward(self.V.vhat)
                self.temp = self.T.backward(self.T.vhat)

            # -- Residual NS
            self.update_NS()

            # -- Ux
            self.update_U(stage=rk)

            # -- Uz
            self.update_V(stage=rk)

            # -- Pressure
            div = self.NS.divergence_velocity(self.U, self.V)
            self.update_P(div)
            self.update_pres(div, stage=rk)

            # -- Update Velocity
            self.update_velocity(self.P, self.U, self.V)

            # -- Temp
            self.update_T(stage=rk)

            # -- Save old velocities
            self.ux_old, self.uz_old, self.temp_old = self.ux, self.uz, self.temp


if __name__ == "__main__":
    shape = (64, 64)
    Pr = 1
    Ra = 5e3

    # Pre-iterate
    NSA = NavierStokesAdjoint(
        shape=shape,
        dt=0.1,
        tsave=1.0,
        Ra=Ra,
        Pr=Pr,
        dealias=True,
        integrator="eu",
        beta=1.0,
    )

    NSA.NS.set_temperature(amplitude=0.2)
    NSA.NS.iterate(5.0)
    U = NSA.NS.U.vhat[:].copy()
    V = NSA.NS.V.vhat[:].copy()
    T = NSA.NS.T.vhat[:].copy()

    # Adjoint iteration
    NSA = NavierStokesAdjoint(
        shape=shape,
        dt=0.1,
        tsave=1.0,
        Ra=Ra,
        Pr=Pr,
        dealias=True,
        integrator="eu",
        beta=1.0,
    )

    NSA.U.vhat[:] = U
    NSA.V.vhat[:] = V
    NSA.T.vhat[:] = T

    NSA.iterate(20.0)
    NSA.plot()
    NSA.animate()

# # Animate
# for i,v in enumerate(NSA.T.V):
#         if NSA.T.V[i][0,0] < 0.1:
#             NSA.T.V[i] += NSA.NS.Tbc.v

# anim = NSA.T.animate(NSA.T.x,duration=4,wireframe=False)
# plt.show()
