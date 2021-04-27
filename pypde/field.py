import numpy as np
from .plot.anim import animate_line,animate_contour,animate_wireframe
from .spectralspace import SpectralSpace

class Field():
    '''
    Class to handle and store field variables
    '''
    def __init__(self,shape):
        self.v = np.zeros(shape) # Field
        self.t = 0               # Time
        self.shape = self.v.shape
        self.dim = len(self.v.shape)

        self.V = []              # Storage Field
        self.T = []              # Storage Time

    def save(self):
        self.V.append(self.v)
        self.T.append(self.t)

    def dstack(self):
        ''' List to ndarray with order [time,space] '''
        self.VS = np.rollaxis( np.dstack(self.V).squeeze(), -1)
        self.TS = np.rollaxis( np.dstack(self.T).squeeze(), -1)

    def check(self):
        return any([
            self.array_is_nan(self.v),
            self.array_is_large(self.v)
            ])

    @staticmethod
    def array_is_large(v,tol=1e10):
        return np.greater(np.abs(v),tol).any()

    @staticmethod
    def array_is_nan(v):
        return np.isnan(v).any()

    def animate(self,x=None,y=None,**kwargs):
        self.dstack()

        if x is None:
            if hasattr(self,"x"):
                x = self.x
            else:
                raise ValueError("Can't animate. x not known.") 

        if self.dim==1:
            return animate_line(x,self.VS,**kwargs)

        if self.dim==2:
            if y is None:
                if hasattr(self,"y"):
                    y = self.y
                else:
                    raise ValueError("Can't animate. y not known.") 

            return animate_wireframe(x,y,self.VS,**kwargs)


class SpectralField(Field,SpectralSpace):
    '''
    Class that contains field variables in real and spectral space
    and how to transform from one to the other.
    
    Input
        shape: int tuple (ndim)
            Shape of field in real space, can be 1d or 2d at the moment
        bases: str tuple
            Define the spectral bases. See Explanation in SpectralSpaces
    '''
    def __init__(self,shape,bases):
        shape,bases = self._check_input(shape,bases)
        assert len(shape) == len(bases), "Shape size must match number of bases"
        Field.__init__(self,shape)
        self._set_bases(bases)
        self.vhat = np.zeros(self.shape_spectral) # Field in spectral space
        
    def _check_input(self,shape,bases):
        if isinstance(shape, int): shape = (shape,)
        if isinstance(bases, str): bases = (bases,)
        return shape, bases
    
    @property
    def x(self):
        return self.xs[0].x
    
    @property
    def y(self):
        if self.dim>1:
            return self.xs[1].x
        else:
            raise ValueError("Dimension y not defined for dim<2.")
            
    def save(self):
        self.backward()
        self.V.append(self.v)
        self.T.append(self.t)
        
    def check(self):
        return any([
            self.array_is_nan(self.vhat),
            self.array_is_large(self.vhat) ])

    def forward(self):
        self.vhat = self.forward_fft(self.v)

    def backward(self):
        self.v = self.backward_fft(self.vhat)