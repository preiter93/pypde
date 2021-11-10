import numpy as np
from pypde import *
from scipy.linalg import eig
import matplotlib.pyplot as plt
from pypde.stability.rbc2d import solve_rbc2d

# Parameters
Nx, Ny = 21, 21
Ra = 1715
aspect = 1.0
Pr = 1.0

# Find the growth rates for given Ra
evals, evecs = solve_rbc2d(
    Nx=Nx, Ny=Ny, Ra=Ra, Pr=Pr, aspect=aspect, norm_diff=True, plot=True
)
