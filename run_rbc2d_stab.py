from pypde import *
from dns import rbc2d
import matplotlib.pyplot as plt

blue = (0 / 255, 137 / 255, 204 / 255)
red = (196 / 255, 0, 96 / 255)
yel = (230 / 255, 159 / 255, 0)

initplot()


def avg_x(f, dx):
    return np.sum(f * dx[:, None], axis=0) / np.sum(dx)


shape = (64, 64)

Pr = 1
Ra = np.logspace(5, 6, 10)
Ra = [5e4]
Nu = []
aspect = 1.0

time = 0.0
e1, e2, e3 = [], [], []
for R in Ra:
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
    NS.set_temperature(m=1, amplitude=0.01)

    N = 5
    dtime = 20.0
    evals, _ = NS.solve_stability(shape=(21, 21), plot=True)
    e3.append(evals[-3])
    e2.append(evals[-2])
    e1.append(evals[-1])

    for i in range(N):
        time += dtime
        NS.iterate(time)
        evals, _ = NS.solve_stability(shape=(21, 21), plot=True)
        e3.append(evals[-3])
        e2.append(evals[-2])
        e1.append(evals[-1])

    fig, ax = plt.subplots()
    i = 0
    for a, b, c in zip(e1, e2, e3):
        alpha = 0.9 - 0.9 / N * i
        ax.scatter(np.real(a), np.imag(a), color=yel, alpha=1 - alpha)
        ax.scatter(np.real(b), np.imag(b), color=red, alpha=1 - alpha)
        ax.scatter(np.real(c), np.imag(c), color=blue, alpha=1 - alpha)
        i += 1
    plt.show()
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
