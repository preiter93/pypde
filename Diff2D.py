import numpy as np 
import matplotlib.pyplot as plt
from pypde.field import SpectralField
from pypde.utils.memoize import memoized
from pypde.solver.matrix import *
from pypde.solver.operator import *
from pypde.solver.base import SolverBase
from pypde.bases import to_sparse

import time
TIMER = np.zeros(2)
class Diffusion2D(SolverBase):
    CONFIG={
        "N": 50,
        "kappa": 1.0,
        "tsave": 0.01,
        "dt": 0.2,
        "ndim": 2,
    }
    def __init__(self,**kwargs):
        SolverBase.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.update_config(**kwargs)
        
        shape = (self.N+2,self.N+2)
        self.field = SpectralField(shape, ("CD","CD"))
        self.time = 0.0              # Time

        # -- Matrices ------
        S = self.field.xs[0].S
        B = self.field.xs[0].B(2)@S
        I = self.field.xs[0].I()@S
        lam = self.dt*self.kappa
        self.Ax = B - lam*I  # lhs
        self.Bx = B          # rhs

        S = self.field.xs[1].S
        B = self.field.xs[1].B(2)@S
        I = self.field.xs[1].I()@S
        lam = self.dt*self.kappa
        self.Ay = B - lam*I  # lhs
        self.By = B          # rhs
    
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
    def y(self):
        return self.field.y

    @property
    @memoized
    def LHS(self):
        '''
        (I-alpha*dt*D2) u = rhs
        '''
        A = self.Ax
        Ax = MatrixLHS(A,ndim=self.ndim,axis=0,
            solver="fdma")
        A = self.Ay
        Ay = MatrixLHS(A,ndim=self.ndim,axis=1,
            solver="fdma")
        
        LHS = LHSImplicit(Ax)
        LHS.add(Ay)
        return LHS
    
    @property
    @memoized
    def RHS(self):
        '''
        lhs = dt*f + u
        only dt*f is stored initially, u is updated in update()
        '''
        fhat = self.field.forward_fft(self._f())
        fhat *= self.dt
        
        b = RHSExplicit(f=fhat)
        b.add_PM(MatrixRHS(self.Bx,axis=0))
        b.add_PM(MatrixRHS(self.By,axis=1))
        return b
        
    def _f(self):
        ''' Forcing Functions'''
        xx,yy = np.meshgrid(self.x,self.y,indexing="ij")
        return np.cos(1*np.pi/2*xx)*np.cos(1*yy)
        
    def update_config(self,**kwargs):
        self.__dict__.update(**kwargs)
        
    def set_bc(self):
        #self.v[ [0],: ] = 0.0 # For neumann
        #self.v[ :,[0] ] = 0.0
        pass
        
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

D = Diffusion2D(N=20,dt=0.01,tsave=0.05)
#D.update()
st=time.perf_counter()
D.iterate(1.0)
en=time.perf_counter()

print("Elapsed time:")
print("RHS:   {:5.4f}".format(TIMER[0]))
print("LHS:   {:5.4f}".format(TIMER[1]))
print("Total: {:5.4f}".format(en-st))

anim = D.field.animate(D.x,D.y,duration=4)
plt.show()


