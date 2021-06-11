import sys

sys.path.append("../")
import numpy as np
from pypde import *
from navier import rbc2d
from stabdict import StabDict, fname_from_Ra, residual, mirror
from adjoint import adjoint
import matplotlib.pyplot as plt
from pathlib import Path

ADJOINT = False

folder = "adiabatic/"
Path(folder).mkdir(parents=True, exist_ok=True)
# Store Results in Dictionary
Ra_dict = StabDict(fname=folder + "adiabatic.txt")

# Default Navier Stokes Settings
ns_settings = {
    "adiabatic": True,
    "shape": (196, 196),
    #"shape": (64, 64),
    "dt": 0.1,
    "tsave": 5.0,
    "Pr": 1.0,
    "dealias": True,
    "integrator": "rk3",
    "beta": 1.0,
    "aspect": 1.0,
}

# Default Newton-LGMRES Settings
ne_settings = {
    "maxiter": 1200,
    "tol": 1e-9,
    "jac_options": {"inner_maxiter": 30},
}

# Default Stability Settings
st_settings = {
    "shape": (25, 25),
    "plot": False,
}


# Set Ra - limits
Ra_lim = [3.3e3, 2e6]

# Initial Save
Ra_dict.save()
Ra_dict.load()

# --------- Presimulate if necessary ----------

# Get Ra_start
for Ra in Ra_dict.dict:
    if Ra > Ra_lim[0]:
        break


NS = rbc2d.NavierStokes(
    Ra=Ra,
    **ns_settings,
)

if not Ra_dict.dict[Ra]:
    print("Find good starting point first ...")
    NS.set_temperature(amplitude=0.04)

    # Simulate
    NS.iterate(100)
    NS.plot()
    Nu, Nuv = NS.eval_Nu()

    # Adjoint
    if ADJOINT: adjoint(NS)

    # Steady State
    sol = NS.solve_steady_state(X0=None, **ne_settings)
    NS.plot()
    Nu, Nuv = NS.eval_Nu()

    # Stability analysis
    evals, evecs = NS.solve_stability(**st_settings)

def mirror(NS):
    #NS.plot()
 #   b = input("Mirror? Y = yes\n")
 #   if b.lower() == "y":
#	NS.U.v[:,:] = -NS.U.v[::-1,:]
#	NS.V.v[:,:] = NS.V.v[::-1,:]
#	NS.T.v[:,:] = NS.T.v[::-1,:]
#	NS.P.v[:,:] = NS.P.v[::-1,:]
#	NS.U.forward()
#	NS.V.forward()
#	NS.T.forward()
#	NS.P.forward()
#	NS.plot()
#	figname = folder + fname[:-1] + ".png"
#	fig, ax = NS.plot(return_fig=True)
#	print("Save fig: " + figname)
#	fig.savefig(figname)
#	plt.close("all")
#	NS.write(folder + fname, add_time=False)
    if np.sum(NS.T.v[0,:]) > np.sum(NS.T.v[-1,:]):
        print("mirror Ra = {:6.2e}".format(NS.Ra))
        NS.U.v[:,:] = -NS.U.v[::-1,:]
        NS.V.v[:,:] = NS.V.v[::-1,:]
        NS.T.v[:,:] = NS.T.v[::-1,:]
        NS.P.v[:,:] = NS.P.v[::-1,:]
        NS.U.forward()
        NS.V.forward()
        NS.T.forward()
        NS.P.forward()
        figname = folder + fname[:-1] + ".png"
        fig, ax = NS.plot(return_fig=True)
        print("Save fig: " + figname)
        fig.savefig(figname)
        plt.close("all")
        NS.write(folder + fname, add_time=False)
        
# ------------ Explore Ra -------------
X0 = None
for Ra in Ra_dict.dict:
    if Ra < Ra_lim[0] or Ra > Ra_lim[1]:
        continue

    print("*** Ra = {:6.2e} ***".format(Ra))
    fname = fname_from_Ra(Ra)

    if Ra_dict.dict[Ra]:
        print("Ra {:6.2e} already known.".format(Ra))
        # Read
        NS.read(folder + fname, add_time=False)
        #mirror(NS)
        continue

    # Update Parameters
    NS.Ra = Ra
    NS.reset(reset_time=True)

    # Adjoint
    if ADJOINT: adjoint(NS)

    # Steady State
    sol = NS.solve_steady_state(
        X0=X0,
        # maxiter=4,
        **ne_settings,
    )
    if ADJOINT: 
        X0 = None
    else:
        X0 = sol.x
    res = residual(sol)

    # Get Nu
    fig, ax = NS.plot(return_fig=True)
    figname = folder + fname[:-1] + ".png"
    print("Save fig: " + figname)
    fig.savefig(figname)
    plt.close("all")
    Nu, Nuv = NS.eval_Nu()

    # Stabilty analysis
    evals, evecs = NS.solve_stability(**st_settings)
    sigma = np.imag(evals[-1])

    # Add to dict
    Ra_dict.add(Ra, Nu, Nuv, sigma, res, fname)

    # Write to file
    NS.write(folder + fname, add_time=False)

    # Write dict
    Ra_dict.save(enforce_overwrite=True)
