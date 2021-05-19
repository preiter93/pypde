import matplotlib.pyplot as plt

def initplot():
    InitPlot()

def update_rc(params):
    """ Update rc params

    Args:
        params (dict): Dictionary containing new settings.
    """
    plt.rcParams.update(params)

#def gfcmap(filename="/home/philipp/Python/plotgf/plotgf/gfcmap.json"):

def gfcmap(filename="gfcmap.json"):
    """"
    Read goldfish colormap

    Args:
        filename (str): Filename (.json) of goldfish colormap
    """
    import json
    import os 
    from matplotlib.colors import LinearSegmentedColormap
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/" + filename, 'r') as fp:
        gfcextdict = json.load(fp)
    return LinearSegmentedColormap('goldfishext', gfcextdict)

class InitPlot:
    """Class managing plot settings
    """
    def __init__(self):
        """ Initialize standard plot settings, color cycle and colormap
        """
        self.plot    = self.default_config()
        self.contour = self.default_config()
        self.contour = self.contour_config(self.contour)

        print("set color cycle ...")
        self.set_color_cycle()
        print("register goldfish colorbar as 'gfcmap' ...")
        self.set_gfcmap()
        print("update rc params to default ...")
        update_rc(self.default_config())

    def default_config(self,size=18.5):
        """ Define default plot settings 

        Args:
            size (int): Fontsize
        """
        config = {'figure.figsize': (4,3),
                  'figure.autolayout': True,
                  'axes.labelsize': size,
                  'axes.titlesize': size,
                  'axes.titlepad': 5,
                  'axes.labelpad': 10,
                  #'axes.linewidth':0.2,
                  'xtick.labelsize': size*0.95,
                  'ytick.labelsize': size*0.95,
                  'lines.linewidth': 1.2,
                  'text.usetex': True,
                  'font.family': 'serif',
                  'font.serif': 'computer modern roman',
                  'savefig.bbox': 'tight',
                  'legend.edgecolor': '0.0',
                  'legend.handlelength': 1.0,
                  'legend.loc': 'upper right',
                  'legend.fontsize': size*0.95,
                  'image.cmap': 'gfcmap',
                  }
        return config

    def contour_config(self, dic):
        add = {
            'figure.figsize': (3, 2),
            'image.cmap': 'gfcmap',
        }
        dic.update(add)
        return dic

    def show_rcparams(self):
        """ Show rcparams keys
        """
        print(plt.rcParams.keys())

    def show(self):
        """ Print plot and contour config
        """
        print("------ Config.plot --------")
        self.show_params(self.plot)
        print("----- Config.contour ------")
        self.show_params(self.contour)

    def show_params(self,dictionary):
        """ Print config parameters

        Args:
            dictionary (dict): Target dictionary
        """
        for keys,values in dictionary.items():
            print("'{:<15s}': {:<20s}".format(keys, str(values)))
        print("")

    def set_color_cycle(self):
        """ Set user defined color cycle
        """
        import matplotlib as mpl
        c1 = (0,0,0) # black
        c2 = (196/255, 0, 96/255) # gfred3
        c3 = (0/255, 137/255, 204/255) # gfblue3
        c4 = (230/255,159/255,0) # yellow
        c5 = (0,158/255,115/255) # green
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
            color=[c1,c2,c3,c4,c5]) 

    def set_gfcmap(self):
        """ Register goldfish colormap
        """
        plt.cm.register_cmap(name='gfcmap', cmap=gfcmap())