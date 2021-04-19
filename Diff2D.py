import numpy as np 
import matplotlib.pyplot as plt
from pypde.field import Field
from pypde.utils.memoize import memoized
from pypde.bases.chebyshev import Chebyshev, ChebDirichlet
from pypde.solver.matrix import *
from pypde.solver.operator import *
from pypde.solver.base import SolverBase
        

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
        
        shape = (self.N,self.N)
        self.field = Field(shape)   # Field variable
        self.time = 0.0              # Time
        self.xf = ChebDirichlet(self.N+2,bc=(0,0))   # Basis in x
        self.x  = self.xf.x          # X-coordinates
    
    @property
    def v(self):
        '''  Main variable (dv/dt) '''
        return self.field.v
    
    @v.setter
    def v(self,value):
        self.field.v = value
        
    @property
    @memoized
    def LHS(self):
        '''
        (I-alpha*dt*D2) u = rhs
        '''
        D2 = self.xf.stiff.toarray()
        M  = self.xf.mass.toarray()
        A = M - self.dt*(self.kappa*D2)
        A1 = MatrixLHS(A,ndim=self.ndim,axis=0,solver="uptria2")
        LHS = LHSImplicit(A1)
        A2 = MatrixLHS(A,ndim=self.ndim,axis=1,solver="uptria2")
        LHS.add(A2)
        return LHS
    
    @property
    @memoized
    def RHS(self):
        '''
        lhs = dt*f + u
        only dt*f is stored initially, u is updated in update()
        '''
        M  = self.xf.mass.toarray()
        fhat = self._f()
        fhat = self.xf.forward_fft(self._f())
        fhat = self.xf.forward_fft(fhat.T).T
        fhat *= self.dt
        
        b = RHSExplicit(f=fhat)
        b.add_PM(MatrixRHS(M,axis=0))
        b.add_PM(MatrixRHS(M,axis=1))
        return b
        
    def _f(self):
        ''' Forcing Functions'''
        xx,yy = np.meshgrid(self.x,self.x)
        return np.cos(1*np.pi/2*xx)*np.cos(1*np.pi/2*yy)
        
    def update_config(self,**kwargs):
        self.__dict__.update(**kwargs)
        
    def set_bc(self):
        self.v[ [-2,-1] ] = 0.0
        
    def update(self):
        ''' 
        Update pde by 1 timestep 
        '''
        self.set_bc()
        self.RHS.b = self.v
        self.v = self.LHS.solve(self.RHS.rhs)
        self.update_time()
        

D = Diffusion2D(N=40,dt=0.01,tsave=0.05)
#D.update()
D.iterate(2.0)

# Transfer stored fields to real space
for i,vv in enumerate(D.field.V):
    D1 = D.xf.backward_fft(vv)
    D.field.V[i] = D.xf.backward_fft(D1.T).T

anim = D.field.animate(D.x,duration=4)
plt.show()