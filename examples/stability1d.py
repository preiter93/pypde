import numpy as np
from pypde import *
from scipy.linalg import eig
import matplotlib.pyplot as plt
from pypde.stability.rbc1d import solve_rbc1d

# Parameters
Ny = 5
alpha = 3.14
Ra = 1715
Pr = 1.0

# Find the growth rates for given Ra
evals, evecs = solve_rbc1d(Ny=Ny, Ra=Ra, Pr=Pr, alpha=alpha, plot=True, norm_diff=True)
