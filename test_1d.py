import numpy as np 
import matplotlib.pyplot as plt
#from pypde.bases_old import * 
from pypde.bases.chebyshev import Chebyshev, ChebDirichlet
from pypde.field import Field
from pypde.utils.memoize import memoized
from pypde.utils.bcs import set_bc,pad
from pypde.solver import *
from pypde.operator import OperatorImplicit



class Diffusion1D(SolverImplicit):
    CONFIG={
        #"L": 2*np.pi,
        "kappa": 1.0,
        "force_strength": 1.0,
        "BC": "Dirichlet",
        "tsave": 0.01,
        "cfl": 0.4,
    }
    def __init__(self,N,**kwargs):
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        self.N = N 

        self.fields={               # Set all fields involved in
        "v": Field(self.N)}         # PDE
    
        self.t = 0.0                      # Time
        self.xf = ChebDirichlet(self.N+2)   # Basis in x
        self.dt = self.cfl_(self.cfl)     # Timestep

        self.sl = self.xf.slice()
        self.I  = self.xf.mass[self.sl,self.sl]
        self.I_inv = self.xf._mass_inv[self.sl,self.sl]
        self.D2 = self.xf.stiff[self.sl,self.sl]

    @property
    def v(self):
        '''  Set main variable dv/dt on which _lhs and _rhs acts '''
        return self.fields["v"].v

    @v.setter
    def v(self,value):
        self.fields["v"].v = value
    
    def _rhs(self):
        ''' Returns rhs of pde. '''
        return self.fhat

    @memoized
    def _lhs(self,dt):
        ''' Returns inverse of the lhs of the pde. 
        Used for implicit calculation. '''
        lhs = self.I_inv@self.D2.toarray()
        lhs = np.eye( self.N) - dt*(self.kappa*lhs)
        return OperatorImplicit(lhs,axis=0,method="inverse")

    @property
    def x(self):
        return self.xf.x

    @property
    def f(self):
        y = np.cos(1*np.pi/2*self.x)
        return y

    @property
    @memoized
    def fhat(self):
        return self.xf.forward_fft(self.f)[self.sl]

    def cfl_(self,safety):
        ''' dt < 0.5 dx**2/kappa'''
        dx = np.min(self.x[1:]-self.x[:-1])
        return 0.5*safety*(dx)**2/self.kappa

    def _set_bc(self):
        self.v[ [-2,-1] ] = 0.0

N = 151
d = Diffusion1D(N,cfl=20.4)
d.update()
plt.spy(d._lhs(0.1)._L)
plt.show()
d.iterate(maxtime=1.0)

# Transfer stored fields to real space
for i,vv in enumerate(d.fields["v"].V):
    d.fields["v"].V[i] = d.xf.backward_fft(vv)

anim = d.fields["v"].animate(d.x,duration=4)
plt.show()