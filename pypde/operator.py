import numpy as np 
from scipy.sparse.linalg import inv as spinv

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

    def solve_inverse(self,rhs):
        rhs = self.premultiply_rhs(rhs)
        if self.axis==0:
            return self.__matmul( self.LI,rhs )
        if self.axis==1:
            return self.__matmul( rhs,self.LI )

    def solve_triangular(self,rhs):
        from scipy.linalg import solve_triangular
        
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