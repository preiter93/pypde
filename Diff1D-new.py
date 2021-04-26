import numpy as np 
import matplotlib.pyplot as plt
from pypde.field import Field,SpectralField
from pypde.utils.memoize import memoized
from pypde.bases.chebyshev import ChebDirichlet,Chebyshev
from pypde.solver.matrix import *
from pypde.solver.operator import *
from pypde.solver.base import SolverBase

import time   
TIMER = np.zeros(2)
class Diffusion1D(SolverBase):
    CONFIG={
        "N": 50,
        "kappa": 1.0,
        "tsave": 0.01,
        "dt": 0.2,
        "ndim": 1,
    }
    def __init__(self,**kwargs):
        SolverBase.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.update_config(**kwargs)
        self.time = 0.0              # Time
        self.field = SpectralField(self.N+2, "ChebDirichlet")

        # -- Matrices ------
        D = Chebyshev(self.N+2).D(2)
        B = Chebyshev(self.N+2).B(2)
        S = self.field.xs[0].S
        self.M =  (B@S)[2:,:]
        self.D2 = (B@D@S)[2:,:]
        
    @property
    def v(self):
        '''  Main variable (dv/dt) '''
        return self.field.vhat
    
    @v.setter
    def v(self,value):
        self.field.vhat = value

    @property
    def x(self):
        return self.field.x
    
    @property
    @memoized
    def LHS(self):
        '''
        (I-alpha*dt*D2) u = rhs
        '''
        A = self.M - self.dt*(self.kappa*self.D2)
        A = MatrixLHS(A,ndim=self.ndim,axis=0,
            solver="fdma")
        return LHSImplicit(A)
    
    @property
    @memoized
    def RHS(self):
        '''
        lhs = dt*f + u
        only dt*f is stored initially, u is updated in update()
        '''

        fhat = self.dt*self.field.forward(self._f())
        b = RHSExplicit(f=fhat)
        b.add_PM(MatrixRHS(self.M,axis=0))
        return b
        
    def _f(self):
        ''' Forcing Functions'''
        return np.cos(1*np.pi/2*self.x)
        
    def update_config(self,**kwargs):
        self.__dict__.update(**kwargs)
        
    def set_bc(self):
        self.v[ [-2,-1] ] = 0.0
        
    def update(self):
        ''' 
        Update pde by 1 timestep 
        '''
        self.set_bc()
        tic = time.perf_counter()
        self.RHS.b = self.v
        rhs = self.RHS.rhs
        toc = time.perf_counter()
        TIMER[0]+=toc-tic
        self.v = self.LHS.solve(rhs)
        tic = time.perf_counter()
        TIMER[1]+=tic-toc
        self.update_time()

D = Diffusion1D(N=50,dt=0.001,tsave=0.1)
# L = D.LHS.A[0].A
# plt.spy(L)
# plt.show()
#D.update()

st=time.perf_counter()
D.iterate(10)
en=time.perf_counter()

print("Elapsed time:")
print("RHS:   {:5.4f}".format(TIMER[0]))
print("LHS:   {:5.4f}".format(TIMER[1]))
print("Total: {:5.4f}".format(en-st))

anim = D.field.animate(D.x,duration=4)
plt.show()