import sys

sys.path.append("./")
from pypde import *
import matplotlib.pyplot as plt


def nu(Ra, Pr, L):
    return np.sqrt(Pr / (Ra / L ** 3.0))


def kappa(Ra, Pr, L):
    return np.sqrt(1 / Pr / (Ra / L ** 3.0))


def Ra(nu, kappa, L):
    return 1 / (nu * kappa) * L ** 3.0


def Pr(nu, kappa):
    return nu / kappa


class NavierStokesBase:
    """
    Some Base Functions for Navier--Stokes Simulations
    """

    def __init__(self, **kwargs):

        self.CONFIG = {
            "shape": (50, 50),
            "Ra": 5e3,
            "Pr": 1.0,
            "dt": 0.2,
            "ndim": 2,
            "tsave": 0.1,
            "dealias": True,
            "integrator": "eu",
            "beta": 1.0,
            "aspect": 1.0,
        }
        self.CONFIG.update(**kwargs)
        self.__dict__.update(**self.CONFIG)
        # self.__dict__.update(**kwargs)

        self.normalize = True
        self.set_nu_kappa()

        # Space for derivatives
        self.deriv_field = Field(
            [
                Base(self.shape[0], "CH", dealias=3 / 2),
                Base(self.shape[1], "CH", dealias=3 / 2),
            ]
        )

        # Coordinates
        self.x = self.deriv_field.x * self.scale[0]
        self.y = self.deriv_field.y * self.scale[1]
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")

        # self.io_config()

    def set_nu_kappa(self, normalize=None):
        if normalize is None:
            normalize = self.normalize

        if normalize:
            self.nu = nu(self.Ra, self.Pr, L=1.0)
            self.kappa = kappa(self.Ra, self.Pr, L=1.0)
            # Scale Physical domain size
            self.scale = (self.aspect * 0.5, 0.5)
        else:
            self.nu = nu(self.Ra, self.Pr, L=2.0)
            self.kappa = kappa(self.Ra, self.Pr, L=2.0)
            # Scale Physical domain size
            self.scale = (self.aspect * 1.0, 1.0)

    def grad(self, field, deriv, return_field=False):
        return grad(field, deriv=deriv, return_field=return_field, scale=self.scale)

    def set_timestep_coefficients_rk3(self):

        """
        (1-a_k*L) phi_k = phi_k + b_k*N_k + c_k * N_k-1

        (Diffusion purely implicit)

        RK3:
            a/dt    b/dt    c/dt
            8/15    8/15    0
            2/15    5/12  -17/60
            1/3     3/4    -5/12
        """
        self.nstage = 3
        self.a = np.array([8.0 / 15.0, 2.0 / 15.0, 1.0 / 3.0])
        self.b = np.array([8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0])
        self.c = np.array([0, -17.0 / 60.0, -5.0 / 12.0])

    def set_timestep_coefficients_euler(self):

        """
        (1-a_k*L) phi_k = phi_k + b_k*N_k + c_k * N_k-1

        (Diffusion purely implicit)
        """
        self.nstage = 1
        self.a = np.array([1.0])
        self.b = np.array([1.0])
        self.c = np.array([0])

    def io_config(self):
        print("----------------------------")
        print("Input Parameter:")
        for k, v in self.CONFIG.items():
            print(k, ":", v)
        print("----------------------------")

    # --- Post processing ----

    def callback(self):
        print(
            "Divergence: {:4.2e}".format(
                np.linalg.norm(self.divergence_velocity(self.U, self.V))
            )
        )

    def plot(self, skip=None, return_fig=False, quiver=False, stream=True):
        # -- Plot
        self.T.backward()
        self.U.backward()
        self.V.backward()

        fig, ax = plt.subplots()
        ax.contourf(
            self.xx, self.yy, self.T.v + self.Tbc.v, levels=np.linspace(-0.5, 0.5, 40)
        )
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_ylim(self.y.min(), self.y.max())

        if quiver:
            # Quiver
            speed = np.max(np.sqrt(self.U.v ** 2 + self.V.v ** 2))
            if skip is None:
                skip = self.shape[0] // 16
            ax.quiver(
                self.xx[::skip, ::skip],
                self.yy[::skip, ::skip],
                self.U.v[::skip, ::skip] / speed,
                self.V.v[::skip, ::skip] / speed,
                scale=7.9,
                width=0.007,
                alpha=0.5,
                headwidth=4,
            )
        if stream:
            from scipy import interpolate

            nx, ny = 41, 41
            x, y = self.x, self.y
            xi = np.linspace(x.min(), x.max(), nx)
            yi = np.linspace(y.min(), y.max(), ny)

            f = interpolate.interp2d(x, y, self.U.v.T, kind="cubic")
            ui = f(xi, yi)
            f = interpolate.interp2d(x, y, self.V.v.T, kind="cubic")
            vi = f(xi, yi)

            speed = np.sqrt(ui * ui + vi * vi)
            lw = 0.8 * speed / np.abs(speed).max()
            ax.streamplot(xi, yi, ui, vi, density=0.75, color="k", linewidth=lw)
        if return_fig:
            return fig, ax
        plt.show()

    def animate(self):
        #  Add inhomogeneous part
        T2 = Field(self.T.xs)
        T2.T = self.T.T
        for i, V in enumerate(self.T.V):
            T2.V.append(V + self.Tbc.v)

        anim = T2.animate(duration=4, x=self.x, y=self.y)
        anim.save("out/anim.gif", writer="imagemagick", fps=20)
        return anim
        # plt.show()

    def eval_Nu(self):
        # from pypde.field_operations import eval_Nu,eval_Nuvol
        Lz = self.y[-1] - self.y[0]
        Nuz = eval_Nu(self.T, self.deriv_field, Tbc=self.Tbc, Lz=Lz)
        Nuv = eval_Nuvol(
            self.T, self.V, self.kappa, self.deriv_field, Tbc=self.Tbc, Lz=Lz
        )
        return Nuz, Nuv

    def interpolate(self, NS_old, spectral=True):
        self.field.interpolate(NS_old.field)

    def write(self, filename=None, leading_str="", add_time=True):
        dict = {
            "nu": self.nu,
            "kappa": self.kappa,
            "Ra": Ra(self.nu, self.kappa, L=self.y[-1] - self.y[0]),
            "Pr": Pr(self.nu, self.kappa),
        }
        self.field.write(
            filename=filename, leading_str=leading_str, add_time=add_time, dict=dict
        )

    def read(self, filename=None, leading_str="", add_time=True):
        dict = {"nu": self.nu, "kappa": self.kappa}
        self.field.read(
            filename=filename, leading_str=leading_str, add_time=add_time, dict=dict
        )
        self.time = self.field.fields[0].t  # Update time
        dict["Ra"] = Ra(dict["nu"], dict["kappa"], L=self.y[-1] - self.y[0])
        dict["Pr"] = Pr(dict["nu"], dict["kappa"])
        self.CONFIG.update(dict)
        self.__dict__.update(**self.CONFIG)
        self.setup_solver()

    def read_from_filename(self, filename):
        pass

    def write_from_Ra(self, folder=""):
        if folder and folder[-1] != "/":
            folder = folder + "/"
        filename = folder + self.fname_from_Ra(self.Ra)
        self.write(filename=filename)

    def read_from_Ra(self, folder=""):
        if folder and folder[-1] != "/":
            folder = folder + "/"
        filename = folder + self.fname_from_Ra(self.Ra)
        self.read(filename=filename)

    @staticmethod
    def fname_from_Ra(Ra):
        """
        Generate filename for given Ra.

        Example:
        fname = fname_from_Ra(Ra)
        NS.write(leading_str=fname,add_time=False)
        """
        return "Flow_Ra{:3.3e}.h5".format(Ra)

    def save(self):
        self.field.save()


class NavierStokesSteadyState:
    """
    Add on for Navier-Stokes class.
    Calculate steaday state solutions using the LGMRES algorithm.
    """

    def solve_steady_state(
        self,
        X0=None,
        dt=None,
        maxiter=300,
        disp=True,
        tol=1e-8,
        jac_options={"inner_maxiter": 30},
    ):
        """
        Solve steady state using scipy's LGMRES algorithm
        """
        from scipy import optimize

        print("\nSolve steady state ...\n")

        """ Solve steady state """
        options = {
            "maxiter": maxiter,
            "disp": disp,
            "fatol": tol,
            "jac_options": jac_options,
        }
        if X0 is None:
            X0 = self.vectorify()

        sol = optimize.root(
            self.steady_fun, X0, args=(self, dt), method="krylov", options=options
        )
        return sol

    def flatten(self):
        return (
            self.T.vhat.flatten().copy(),
            self.U.vhat.flatten().copy(),
            self.V.vhat.flatten().copy(),
        )

    def reshape(self, X):
        T_mask, U_mask, V_mask = self.get_masks()
        That = X[T_mask].copy().reshape(self.T.vhat.shape)
        Uhat = X[U_mask].copy().reshape(self.U.vhat.shape)
        Vhat = X[V_mask].copy().reshape(self.V.vhat.shape)
        return That, Uhat, Vhat

    def vectorify(self):
        return np.concatenate((self.flatten()))

    def get_masks(self):
        t, u, v = self.flatten()
        T_mask = slice(0, t.size)
        U_mask = slice(t.size, t.size + u.size)
        V_mask = slice(t.size + u.size, t.size + u.size + v.size)
        return T_mask, U_mask, V_mask

    def steady_fun(self, X, NS, dt):
        """
        Input:
            X: ndarray (1D)
                Flow field vector [T,u,v]

        Output
            ndarry (1D)
                Residual vector [Tr,ur,v]
        """
        NS.T.vhat[:], NS.U.vhat[:], NS.V.vhat[:] = NS.reshape(X)
        if dt is None:
            dt = NS.dt
            NS.update()
        else:
            NS.reset_time()
            NS.iterate(dt, callback=False)
            NS.reset_time()

        Y = NS.vectorify()
        return (Y - X) / dt


def eval_Nu(T, field, Lz=1.0, Tbc=None):
    """
    Heat Flux at the plates
    """
    T.backward()
    T = T.v.copy()
    if Tbc is not None:
        T += Tbc.v.copy()

    That = field.forward(T)
    scale = Lz / 2.0
    dThat = field.derivative(That, 1, axis=1) / scale
    dT = field.backward(dThat)

    dTavg = avg_x(dT, field.dx)
    Nu_bot = -dTavg[0] * Lz
    Nu_top = -dTavg[-1] * Lz
    print("Nubot: {:10.6e}".format(Nu_bot))
    print("Nutop: {:10.6e}".format(Nu_top))
    return (Nu_bot + Nu_top) / 2.0


def eval_Nuvol(T, V, kappa, field, Lz=1.0, Tbc=None):
    """
    Heat Flux through the box (volume)
    """
    T.backward()
    V.backward()

    T = T.v.copy()
    if Tbc is not None:
        T += Tbc.v.copy()
    V = V.v.copy()

    That = field.forward(T)
    scale = Lz / 2.0
    dThat = field.derivative(That, 1, axis=1) / scale
    dT = field.backward(dThat)

    Nuvol = (T * V / kappa - dT) * Lz
    Nuvol = avg_vol(Nuvol, field.dx, field.dy)
    print("Nuvol: {:10.6e}".format(Nuvol))
    return Nuvol


class NavierStokesStability:
    """
    Add on for Navier-Stokes class.
    Conduct linear stability analysis on a given flow
    """

    def solve_stability(self, shape=(21, 21), plot=True, n_evecs=3):
        from pypde.stability.utils import print_evals
        from pypde.stability.rbc2d import solve_stability_2d, plot_evec
        from pypde.field_operations import interpolate

        print("\nSolve stability ...\n")

        # Initialize Navier Stokes class on coarser grid
        config = self.CONFIG.copy()
        config["shape"] = shape
        self.NS_C = self.__class__(case=self.case, **config)
        # self.NS_C = NavierStokes(adiabatic=self.adiabatic, **config)

        # Interpolate onto coarser grid
        interpolate(self.T, self.NS_C.T, spectral=True)
        interpolate(self.V, self.NS_C.V, spectral=True)
        interpolate(self.U, self.NS_C.U, spectral=True)

        # Set fields
        U = self.NS_C.U
        V = self.NS_C.V
        T = self.NS_C.T
        P = self.NS_C.P
        CH = Field([Base(shape[0], "CH", dealias=2), Base(shape[1], "CH", dealias=2)])

        # Extract flow fields
        UU = galerkin_to_cheby(self.NS_C.U.vhat, U)
        VV = galerkin_to_cheby(self.NS_C.V.vhat, V)
        TT = galerkin_to_cheby(self.NS_C.T.vhat, T)
        TT += galerkin_to_cheby(self.NS_C.Tbc.vhat, self.NS_C.Tbc)
        # temp = CH.backward(TT)

        # plt.contourf(self.NS_C.xx, self.NS_C.yy, temp)
        # plt.show()

        # Calculate stability
        evals, evecs = solve_stability_2d(
            self.nu,
            self.kappa,
            1.0,
            U,
            V,
            T,
            P,
            CH,
            UU,
            VV,
            TT,
            scale=self.scale,
            # norm_diff=False,
        )
        print_evals(evals, 5)
        if plot:
            xx, yy = self.NS_C.xx, self.NS_C.yy
            for i in range(n_evecs):
                plot_evec(evecs, U, V, P, T, xx, yy, m=-i - 1)
        print("Stability calculation finished!")
        return evals, evecs
