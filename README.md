# pypde
Partial differential equations solver in python.
<img align="right" src="https://www.python.org/static/community_logos/python-logo-generic.svg" width="150">

## Method
Spectral Galerkin

## Implemented basis functions
Chebyshev  
Chebyshev - Dirichlet [[1]](#1)  
Chebyshev - Neumann [[1]](#1)

## Rayleigh-Benard Convection
The code is developed for the purpose to study Rayleigh-Benard Convection. 
It implements the coupled Navier-Stokes momentum and temperature equations. 

## Dependencies

python>=3.6  
pyFFTW  
h5py  

## Installation of pyFFTW

The pyFFTW version 0.12.0 does not support dct's, it is necessary to install
pyFFTW from the newest repository:

pip3 install git+https://github.com/pyFFTW/pyFFTW.git

## References
<a id="1">[1]</a> 
Shen, J. (1995). 
Effcient Spectral-Galerkin Method II. Direct Solvers of Second and Fourth Order Equations by Using Chebyshev Polynomials. 
Siam J. Sci. Comput., Vol. 16, No. 1, pp. 74-87.
