import numpy as np 
import matplotlib.pyplot as plt
from pypde.bases_old import * 
from pypde.field import Field
from pypde.utils.memoize import memoized
from pypde.solver_old import *
from pypde.operator import OperatorImplicit


class Diffusion2D(SolverImplicit):
    CONFIG={
        "L": 2*np.pi,
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
        self.shape = (N,N) 

        self.fields={                 # Set all fields involved in
        "v": Field(self.shape)}       # PDE

        self.t = 0.0                  # Time
        self.xf = Chebyshev(self.N)   # Basis in x
        self.dt = self.cfl_(self.cfl) # Timestep

    @property
    def v(self):
        '''  Set main variable dv/dt on which _lhs and _rhs acts '''
        return self.fields["v"].v

    @v.setter
    def v(self,value):
        self.fields["v"].v = value

    def _rhs(self):
        ''' Returns rhs of pde. Used for explicit calculation. '''
        dv = self.xf.deriv_dm( self.fields["v"].v, 2,axis=0 )
        dv+= self.xf.deriv_dm( self.fields["v"].v, 2,axis=1 )
        return self.f                   # Fully implicit
        #return self.f + self.kappa*dv    # Fully explicit 

    @memoized
    def _lhs(self,dt):
        ''' Returns inverse of the lhs of the pde. 
        Used for implicit calculation. '''
        L = np.eye(self.N)- dt*(self.kappa*
            self.xf.get_deriv_mat(2))
        self.xf.set_bc(L,pos=[0,-1],which=self.BC)
        return [OperatorImplicit(L,axis=0),
        OperatorImplicit(L,axis=1)]

    @property
    def x(self):
        return self.xf.x

    #@property
    #def f(self):
    #    ''' Define forcing '''
    #    pos = self.N*2//4
    #    f=np.zeros(self.shape)
    #    f[ pos, pos] += self.force_strength
    #    #f[-pos, :] -= self.force_strength
    #    return f
    @property
    def f(self):
        ''' Forcing Functions'''
        xx,yy = np.meshgrid(self.x,self.x)
        return np.cos(1*np.pi/2*xx)*np.cos(1*np.pi/2*yy)

    def cfl_(self,safety):
        ''' dt < 0.5 dx**2/kappa'''
        dx = np.min(self.x[1:]-self.x[:-1])
        return 0.5*safety*(dx)**2/self.kappa

    def _set_bc(self):
        self.v[ [0,-1], : ] = 0.0
        self.v[ :,  [0,-1]] = 0.0

N = 30
d = Diffusion2D(N,cfl=10.0)
d.iterate(maxtime=1.0)

anim = d.fields["v"].animate(d.x,d.x)
plt.show()