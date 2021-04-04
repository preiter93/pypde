import numpy as np
from .plot.anim import animate_line,animate_contour,animate_wireframe

class Field():
    '''
    Class to handle and store field variables
    '''
    def __init__(self,shape):
    	self.v = np.zeros(shape) # Field
    	self.t = 0 			     # Time
    	self.shape = self.v.shape
    	self.dim = len(self.v.shape)

    	self.V = []              # Storage Field
    	self.T = []              # Storage Time

    def save(self):
        if hasattr(self, "v_hat"):
            self.v = np.real( np.fft.ifft(self.v_hat) )
        self.V.append(self.v)
        self.T.append(self.t)

    def dstack(self):
        ''' List to ndarray with order [time,space] '''
        self.VS = np.rollaxis( np.dstack(self.V).squeeze(), -1)
        self.TS = np.rollaxis( np.dstack(self.T).squeeze(), -1)

    def check(self):
    	return any([
    		self.array_is_nan(),
    		self.array_is_large()
    		])

    def array_is_large(self,tol=1e10):
    	return np.greater(np.abs(self.v),tol).any()

    def array_is_nan(self):
        return np.isnan(self.v).any()

    def animate(self,x,y=None,**kwargs):
    	self.dstack()
    	if self.dim==1:
    		return animate_line(x,self.VS,**kwargs)
    	if self.dim==2:
    		if y is None: y=x
    		return animate_wireframe(x,y,self.VS,**kwargs)


