import numpy as np 


class OperatorImplicit():
    ''' 
    Class that contains the lhs matrix
    for implicit solvers
    '''
    def __init__(self,L,axis=0):
        self.axis = axis
        self._L  = L
        self._LI = self.__inv(self.L)

    def solve(self,rhs):
        if self.axis==0:
            return self.__matmul( self.LI,rhs )
        if self.axis==1:
            return self.__matmul( rhs,self.LI )

    @property
    def L(self):
        return self._L

    @property
    def LI(self):
        return self._LI

    @L.setter
    def L(self,value):
        self._L = value
        self._LI = self.__inv(self.L)

    @property
    def LI(self):
        return self._LI
    
    def __matmul(self,A,B):
        ''' Matrix multiplication '''
        if len(A.shape)==2:
            return A@B 
        if len(A.shape)==1:
            return A*B

    def __inv(self,A):
        ''' Invert '''
        if len(A.shape)==2:
            return np.linalg.inv(A)
        if len(A.shape)==1:
            return A**-1