from navier import rbc2d, rbc2d_adj
from pypde import *

initplot()

# Default Navier Stokes Settings
ns_settings = {
    "case": "rbc",
    "shape": (12, 12),
    "dt": 0.02,
    "tsave": 1.0,
    "pr": 1.0,
    "dealias": False,
    "integrator": "eu",
    "beta": 1.0,
    "aspect": 1.0,
}

# Defauld Adjoint settings
nsa_settings = ns_settings.copy()

Ra = 1e4

NS = rbc2d.NavierStokes(
    ra=Ra,
    **ns_settings,
)

NS.set_velocity(m=1, n=1, amplitude=0.2)
NS.iterate(30.0)
NS.plot()
NS.eval_Nu()
NS.write("test.h5")

fname = "test.h5"
# fname = "flow100.000.h5"
NS.read(fname)
NS.eval_Nu()
NS.iterate(50.0)
# NS.plot()

NSA = rbc2d_adj.NavierStokesAdjoint(
    ra=NS.ra,
    **nsa_settings,
)
NSA.read(fname)
NSA.plot()
NSA.reset_time()
NSA.update()
NSA.iterate(10)
# NSA.eval_Nu()

# NS.solve_steady_state()
# NS.solve_stability(shape=(27, 27))
# NS.write_from_Ra()
