import h5py
import numpy as np
import os


def merge_fields(fname_t, folder=""):
    u_field = fname_t.replace("_T", "_U")
    if os.path.isfile(u_field):
        print("U field found.")
    else:
        raise ValueError("U field {:} not found.".format(u_field))

    v_field = fname_t.replace("_T", "_V")
    if os.path.isfile(v_field):
        print("V field found.")
    else:
        raise ValueError("V field {:} not found.".format(v_field))

    scalars = {"Ra": (), "Pr": (), "nu": (), "kappa": (), "time": ()}

    # Read common variables from temperature field
    f = h5py.File(fname_t, "r")
    for s in scalars:
        scalars[s] = np.array(f.get(s))
    scalars["time"] = np.array(0.0)
    print("Ra = {:4.3e}".format(scalars["Ra"]))
    x = np.array(f.get("x"))
    y = np.array(f.get("y"))
    T = np.array(f.get("v"))
    That = np.array(f.get("vhat"))
    f.close()

    # Read velocity fields
    f = h5py.File(u_field, "r")
    U = np.array(f.get("v"))
    Uhat = np.array(f.get("vhat"))
    f.close()

    f = h5py.File(v_field, "r")
    V = np.array(f.get("v"))
    Vhat = np.array(f.get("vhat"))
    f.close()

    # Create new collected field
    fname = fname_t[:-5] + ".h5"
    fname = fname.replace(folder, folder + "Flow_")
    print("Collect T,U,V fields into {:}".format(fname))
    # Write
    hf = h5py.File(fname, "w")
    for s in scalars:
        hf.create_dataset(s, data=scalars[s])
    hf.create_dataset("x", data=y)
    hf.create_dataset("y", data=x)

    # Groups
    for (v, vhat, name) in zip([T, U, V], [That, Uhat, Vhat], ["temp", "ux", "uz"]):
        grp = hf.create_group(name)
        grp["v"] = v
        grp["vhat"] = vhat
    hf.close()
