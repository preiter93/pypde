import numpy as np
import os.path


class StabDict:
    """
    Stores results of the steady state LSC analysis.

    Can be modified, at the moment it holds:
        Nu   : Nu through the plates
        Nuv  : Nu bases on volume heat flux
        sigma: Maximum growth rate of steady state
        res  : Residual of steady state (should be <1e-8)
        fname: HDF5 filename (str)

    Example:
        Ra_all = np.logspace(3,7,4)
        Ra_dict = StabDict(Ra_all)
        Ra = Ra_all[0]
        Ra_dict.add(Ra,1.,1.,2.,4e-3,"ra1e3")
        Ra_dict.save()
        Ra_dict.load()
    """

    def __init__(self, Ra_all=None, fname="test.txt"):
        if Ra_all is None:
            Ra_all = np.logspace(3, 7, 200)
        self.dict = {}
        self.fname = fname

        for Ra in Ra_all:
            self.dict[Ra] = ()

        # for saving
        self.header = "{:6s}|{:6s}|{:6s}|{:6s}|{:6s}|{:14s}".format(
            "Ra", "Nu", "Nuv", "sigma", "res", "fname"
        )
        self.wtype = "f8, f8, f8, f8, f8, S14"
        self.fmt = "%6.2e %6.2f %6.2f %6.2f %6.2e %14.14s"

        # type for reading
        self.rtype = "f8, f8, f8, f8, f8, U14"

    def add(
        self, Ra: float, Nu: float, Nuv: float, sigma: float, res: float, fname: str
    ):
        """
        Example
        Ra = list(Ra_dict.dict.keys())[0]
        Ra_dict.add(Ra,1.,1.,2.,4e-3,"ra1e3")
        """
        if Ra not in self.dict:
            raise ValueError("Can't add Ra. Not known to dict.")
        tpl = (Nu, Nuv, sigma, res, fname)
        if not self.dict[Ra]:
            self.dict[Ra] = tpl
        else:
            print("Key {:} already in Ra_dict!".format(Ra))

    def save(self, fname=None, enforce_overwrite=False):

        if fname is None:
            fname = self.fname

        if not enforce_overwrite:
            if os.path.isfile(fname):
                overwrite = input("File already exists. Overwrite? Y = yes\n")
                if not overwrite.lower() == "y":
                    print("Aborted ...")
                    return
                else:
                    print("Overwrite ...")

        # Create mixed type table
        table = np.zeros((len(self.dict)), dtype=self.wtype)

        # Fill table
        for i, Ra in enumerate(self.dict):
            if self.dict[Ra]:
                table[i] = (Ra, *self.dict[Ra])
            else:
                table[i][0] = Ra

        # Save to file
        np.savetxt(fname, table, fmt=self.fmt, header=self.header)

    def load(self, fname=None):

        if fname is None:
            fname = self.fname

        table = np.loadtxt(fname, dtype=self.rtype)

        # Add to dictionary if fname is not empty
        for tpl in table:
            if not (tpl[-1] == "b''" or tpl[-1] == ""):
                self.add(*tpl)


def fname_from_Ra(Ra):
    """
    Generate leading_str for given Ra.

    Example:
    fname = fname_from_Ra(Ra)
    NS.write(leading_str=fname,add_time=False)
    """
    return "Ra{:4.2e}_".format(Ra)
