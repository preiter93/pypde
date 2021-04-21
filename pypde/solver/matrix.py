import numpy as np 
import scipy.sparse as sp

class MatrixBase():
    ''' 
    Store Scalars/Matrices and Information along which axis they
    should be multiplied.
    
    Input
        A: scalar or ndarray
        axis: int
            Defines along which axis PM should be applied
        
    '''
    scalar = False
    def __init__(self,A,axis=0,sparse=False):
        self.sparse = sparse
        self.axis = axis
        
        # Input is a scalar
        if isinstance(A,float) or isinstance(A,int):
            self.A = float(A)
            self.scalar=True
            
        # Input is a Matrix
        elif not sp.issparse(A) and sparse:
            self.A = sp.csr_matrix(A)
        elif sp.issparse(A) and not sparse:
            self.A = self.A.toarray()
        else:
            self.A = A
            
        
        # Store Transpose if axis ==1
        if self.axis==1 and not self.scalar:
            if self.sparse:
                self.AT = sp.csr_matrix(
                    self.A.toarray().T)
            else:
                self.AT = self.A.T
                
        # Define matrix multiplication
        if self.scalar:
            self.dot = self.dot_sc
        else:
            self.dot = self.dot_mm
            
            
    def dot_mm(self,b):
        if self.axis==0:
            return self.A@b
        else:
            return b@self.AT
        
    def dot_sc(self,b):
        return self.A*b
    
class MatrixRHS(MatrixBase):
    def __init__(self,A,axis=0):
        MatrixBase.__init__(self,A,axis,sparse=True)  

from pypde.solver.fortran import linalg as lafort

class MatrixLHS(MatrixBase):
    '''
    Extends MatrixBase and adds functionality to solve
        Ax = b
    where the type of solver should be chosen by the 
    type of matrix A.
    
    Input:
        See MatrixBase
        
        ndim: int
            Dimensionality of rhs b
            
        axis: int
            Axis along which A should act on b
        
        solver: str (default='solve')
            'solve' : General solver np.linalg.solver(A,b)
            'tdma': A is banded with diagonals in offsets  0, 2
            'fdma': A is banded with diagonals in offsets -2, 0, 2, 4
            
    '''
    all_methods = [
        "solve",
        #"uptria",
        #"uptria2",
        "tdma",
        "fdma"
    ]
    def __init__(self,A,ndim,axis,sparse=False,solver="solve"):
        MatrixBase.__init__(self,A,axis,sparse=sparse)
        self.solver = solver
        self.ndim = ndim
        self.set_subsolver()
        
        assert solver in self.all_methods, "Solver type not found!"
        if solver in ["solve"]:
            self.solve = self.solver_solve
        if solver in ["uptria"]:
            self.solve = self.solve_uptria
        if solver in ["fdma"]:
            L = self.A
            self.d, self.u1 = np.diag(L).copy(),np.diag(L,+2).copy()
            self.solve = self.solve_tdma
        if solver in ["fdma"]:
            L = self.A
            self.d, self.u1 = np.diag(L).copy(),    np.diag(L,+2).copy()
            self.u2, self.l = np.diag(L,+4).copy(), np.diag(L,-2).copy()
            FDMA_LU(self.l, self.d, self.u1, self.u2)
            self.solve = self.solve_fdma

    def solver_solve(self,b):
        ''' 
        Solve Ax=b, where A is of any kind
        '''
        assert self.ndim == b.ndim, "Dimensionality mismatch in MatrixLHS. Check ndim."
        if self.axis==0:
            return np.linalg.solve(self.A,b)
        elif self.axis==1:
            assert b.ndim > 1
            return np.linalg.solve(self.A,b.T)

    def solve_tdma(self,b):
        self._tdma(self.d,self.u1,b,0)
        return b

    def solve_fdma(self,b):
        self._fdma(self.d,self.u1,self.u2,self.l,b,0)
        return b
        
    def set_subsolver(self):
        self._tdma = lafort.tridiagonal.solve_tdma
        self._fdma = lafort.tridiagonal.solve_fdma

    def _lu_decomp(self,A):
        ''' LU Decomposition of A'''
        from scipy.linalg import lu
        P,L,U = lu(A)


def FDMA_LU(ld, d, u1, u2):
        n = d.shape[0]
        for i in range(2, n):
            ld[i-2] = ld[i-2]/d[i-2]
            d[i] = d[i] - ld[i-2]*u1[i-2]
            if i < n-2:
                u1[i] = u1[i] - ld[i-2]*u2[i-2]