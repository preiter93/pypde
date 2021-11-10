from navier import rbc2d, rbc2d_adj
from pypde import *

initplot()

# Default Navier Stokes Settings
ns_settings = {
    "case": "rbc",
    "shape": (64, 64),
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
NS.iterate(4.0)
NS.plot()
NS.eval_Nu()
