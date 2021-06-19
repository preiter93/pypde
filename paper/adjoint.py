import sys

sys.path.append("../")
# import numpy as np
# from pypde import *
from navier import rbc2d_adj


def adjoint(NS, time=100.0):
    config = NS.CONFIG.copy()
    NSA = rbc2d_adj.NavierStokesAdjoint(
        adiabatic=NS.adiabatic, ns_type=NS.__class__, **config
    )
    # NSA.plot()
    # NS -> NSA
    NSA.NS.U.vhat[:] = NS.U.vhat[:].copy()
    NSA.NS.V.vhat[:] = NS.V.vhat[:].copy()
    NSA.NS.T.vhat[:] = NS.T.vhat[:].copy()
    NSA.U.vhat[:] = NS.U.vhat[:].copy()
    NSA.V.vhat[:] = NS.V.vhat[:].copy()
    NSA.T.vhat[:] = NS.T.vhat[:].copy()
    # Iterate
    NSA.iterate(time)
    # NSA -> NS
    NS.U.vhat[:] = NSA.U.vhat[:]
    NS.V.vhat[:] = NSA.V.vhat[:]
    NS.T.vhat[:] = NSA.T.vhat[:]
    # NS.plot()
