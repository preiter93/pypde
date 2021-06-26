from navier import rbc2d
from pypde import *

initplot()

shape = (32, 32)
Pr = 1
Ra = np.logspace(5, 6, 10)
Ra = [5e4]
Nu = []
aspect = 1.0
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
    NS.set_temperature(m=1, amplitude=0.05)
    NS.iterate(10.0)
    NS.plot()
    # NS.solve_steady_state()
    # NS.solve_stability(shape=(27, 27))
    # NS.write_from_Ra()
