import numpy as np 
from scipy.sparse.linalg import inv as spinv
import scipy as sp
from pypde.bases.solver.tdma import solve_twodma
from pypde.bases.solver.utda import solve_triangular

class OperatorImplicit():
    ''' 
    Class that contains the lhs matrix
    for implicit solvers
    '''
    def __init__(self,L,axis=0,method="inverse",rhs_prefactor=None):
        self.axis = axis
        self._L  = L
        self.rhs_pm = rhs_prefactor
    
        if method in ["inv","inverse"]:
            self._LI = self.__inv(self.L)
            self.solve = self.solve_inverse
        if method in ["tma", "triangular"]:
            self.solve = self.solve_triangular
        if method in ["hh", "helmholtz"]:
            self.lu_decomp(self._L)
            self.solve = self.solve_helmholtz_fortran

    def lu_decomp(self,A):
        P,L,U = sp.linalg.lu(A)
        self.Lower, self.Upper = L,U 
        self.d, self.u1 = np.diag(self.Lower), np.diag(self.Lower,-2)

    def solve_inverse(self,rhs):
        rhs = self.premultiply_rhs(rhs)
        if self.axis==0:
            return self.__matmul( self.LI,rhs )
        if self.axis==1:
            return self.__matmul( rhs,self.LI )

    def solve_triangular(self,rhs):
        if self.axis==0:
            rhs = self.premultiply_rhs(rhs)
            return solve_triangular(self.L, rhs,   lower=False)
        if self.axis==1:
            assert rhs.ndim > 1
            rhs = self.premultiply_rhs(rhs.T)
            return solve_triangular(self.L, rhs, lower=False).T


    def premultiply_rhs(self,rhs):
        if self.rhs_pm is not None:
            try:
                return self.rhs_pm@rhs
            except:
                return self.rhs_pm*rhs
        return rhs

    def solve_helmholtz(self,rhs):
        rhs = self.premultiply_rhs(rhs)
        TwoDMA_SolveU(self.d,self.u1,rhs)
        return sp.linalg.solve_triangular(self.Upper,rhs)

    def solve_helmholtz_fortran(self,rhs):
        rhs = self.premultiply_rhs(rhs)
        solve_twodma(self.d,self.u1,rhs,rhs.size)
        return solve_triangular(self.Upper,rhs,rhs.size)



    @property
    def L(self):
        return self._L

    @property
    def LI(self):
        return self._LI

    @L.setter
    def L(self,value):
        self._L = value
        #self._LI = self.__inv(self.L)

    @property
    def LI(self):
        return self._LI
    
    def __matmul(self,A,B):
        ''' Matrix multiplication '''
        if len(A.shape)==2:
            mm = np.array(A)@np.array(B) 
            return mm
        if len(A.shape)==1:
            return A*B

    def __inv(self,A):
        ''' Invert '''
        if len(A.shape)==2:
            try:
                return np.linalg.inv(A)
            except:
                return spinv(A)
        if len(A.shape)==1:
            return A**-1

def TwoDMA_SolveU(d, u1, x, axis=0):
    ''' 
    d: N
        diagonal
    u1: N-2
        Diagonal with offset -2
    x: array ndim==1
        rhs
    '''
    assert x.ndim == 1, "Use optimized version for multidimensional solve"
    n = d.shape[0]
    x[0] = x[0]/d[0]
    x[1] = x[1]/d[1]
    for i in range(2,n - 1):
        x[i] = (x[i] - u1[i-2]*x[i-2])/d[i]