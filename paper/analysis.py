import sys

sys.path.append("../")

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pypde import *
import h5py
from utils import merge_fields

class Flow:
    def __init__(self, folder, **kwargs):
        self.folder = folder
        self.flowlist = list(glob.glob(folder + "Flow*.h5"))
        # Convert individual files to common file if necessary
        if len(self.flowlist) == 0:
            flowlist = list(glob.glob(folder + "Ra*_T.h5"))
            if len(flowlist) == 0:
                raise ValueError("No matching files found in {:}".format("folder"))
            else:
                for f in flowlist:
                    merge_fields(f, folder)
            self.flowlist = list(glob.glob(folder + "Flow*.h5"))

        # Extract Ra
        self.Ra = []
        for f in self.flowlist:
            self.Ra.append(self.get_Ra(f))
        # Sort
        argsort = np.argsort(self.Ra)
        self.flowlist = list(np.array(self.flowlist)[argsort])
        self.Ra = list(np.array(self.Ra)[argsort])

        print("Matching flow files found.")
        for i, f in enumerate(self.flowlist):
            print("{:3d} : {:}".format(i, f))

        # Default Navier Stokes Settings
        self.ns_settings = {
            "shape": (196, 196),
            "dt": 0.1,
            "tsave": 1.0,
            "Pr": 1.0,
            "dealias": True,
            "integrator": "rk3",
            "beta": 1.0,
            "aspect": 1.0,
        }

        self.ns_settings.update(**kwargs)
        print(self.ns_settings)

    def get_Ra(self, f):
        hf = h5py.File(f, "r")
        Ra = np.array(hf.get("Ra"))
        hf.close()
        return Ra

    def get_NS(self, fname):
        from navier import rbc2d

        Ra = self.get_Ra(fname)
        NS = rbc2d.NavierStokes(
            Ra=Ra,
            **self.ns_settings,
        )
        NS.read(filename=fname)
        return NS


class Analyse:
    def __init__(self, NS):
        self.qlist = {}
        self.NS = NS
        self.fmt ='%.5e'

        self.run()

    def run(self):
        self.add_nu()
        self.add_sigma()

    def add_nu(self):
        nu,nuv = self.NS.eval_Nu()
        self.qlist["Nu"] = nu
        self.qlist["Nuv"] = nuv

    def add_sigma(self):
        evals, evecs = self.NS.solve_stability(shape=(27,27), plot = False)
        self.qlist["sigma"] = np.imag(evals[-1])

    def to_array(self):
        return np.array([[self.NS.Ra, *[self.qlist[key] for key in self.qlist]]])

    def header(self):
        return (' '*7).join(["# Ra", *[key for key in self.qlist]])

    def save(self, fname = "qlist.txt"):
        if not os.path.isfile(fname):
            with open(fname, "w") as f:
                f.write(self.header())
            with open(fname, "a") as f:
                f.write(b"\n")
        with open(fname, "a") as f:
            #f.write(b"\n")
            np.savetxt(f, self.to_array(), fmt=self.fmt)


folder = "linear/"
case = "linear"

flow = Flow(folder, case = case)

for f in flow.flowlist[:]:
    NS = flow.get_NS(f)
    print("Ra = {:4.3e}".format(NS.Ra))
    #NS.plot()
    A = Analyse(NS)
    A.save(fname = "qlist_"+case+"txt")

# NS.solve_stability(shape=(27,27))
# NS.eval_Nu()
# NS.solve_steady_state()

# NS.eval_Nu()
# NS.plot()
#NS.write_from_Ra()
#print(NS.Ra)
# NS.plot()
# NS.iterate(2)
# NS.plot()
