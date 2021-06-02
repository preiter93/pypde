from pypde import *
from dns import rbc2d

initplot()


def avg_x(f, dx):
    return np.sum(f * dx[:, None], axis=0) / np.sum(dx)


shape = (32, 32)

Pr = 1
Ra = np.logspace(5, 6, 10)
Ra = [5e4]
Nu = []
aspect = 1.0
for R in Ra:
    # r = R/2**3
    # nu = np.sqrt(Pr/r)
    # kappa = np.sqrt(1/Pr/r)

    # -- Solve Navier Stokes
    NS = rbc2d.NavierStokes(
        shape=shape,
        dt=0.1,
        tsave=1.0,
        Ra=R,
        Pr=Pr,
        dealias=True,
        integrator="rk3",
        beta=1.0,
        aspect=aspect,
    )
    NS.set_temperature(m=1, amplitude=0.05)
    NS.iterate(100.0)
    NS.solve_steady_state()
    NS.solve_stability(shape=(27, 27))
    # NS.write()

    # -- Animate and plot
    NS.plot()
    NS.animate()

    # -- Get Geometry
    x, y = NS.T.x, NS.T.y
    dx, dy = NS.T.dx, NS.T.dy
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # -- Evaluate Nu
    Nuz, Nuv = NS.eval_Nu()
    Nu.append(Nuz)
