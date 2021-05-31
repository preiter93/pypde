# pypde

General light-weight spectral method simulation framework.  

Implemented bases:  
Chebyshev  
Chebyshev-Dirichlet  
Chebyshev-Neumann  
Fourier  

In progress ...

## Dependencies

python>=3.6  
pyFFTW  
h5py  

## Installation of pyFFTW

The pyFFTW version 0.12.0 does not support dct's, it is necessary to install
pyFFTW from the newest repository:

pip3 install git+https://github.com/pyFFTW/pyFFTW.git
