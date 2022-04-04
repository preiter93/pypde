import numpy as np
from pypde import *
from scipy.linalg import eig
import matplotlib.pyplot as plt
from pypde.stability.rbc1d import solve_diff1d

# Parameters
Ny = 50

# Find the growth rates for given Ra
evals, evecs = solve_diff1d(Ny=Ny, kappa=1)
