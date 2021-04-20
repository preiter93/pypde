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


from scipy.linalg import solve_triangular as sp_triangular
from pypde.solver.fortran import linalg as lafort
from pypde.bases.solver.tdma import solve_twodma

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
            'diag'  :  A is filled only on the main diagonal
            'uptria':  A is upper triangular
            'uptria2': A is upper triangular with 1 lower
                subdiagonal shifted by 2. Arising in HelmholtzProblem
            
    '''
    all_methods = [
        "solve",
        "uptria",
        "uptria2",
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
        if solver in ["uptria2"]:
            L,U = self._lu_decomp(self.A)
            self.U = U
            self.d, self.u1 = np.diag(L), np.diag(L,-2)
            self.solve = self.solve_uptria2
        

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
        
    def solve_uptria(self,b):
        ''' 
        Solve Ax=b, where A is upper triangular
        '''
        if self.axis==1:
            assert b.ndim > 1

        return self._triangular(self.A, b,self.axis)

    def solve_uptria2(self,b):
        ''' 
        Solve Ax=b, where A is triangular with 
        only 1 subdiagonal.
            | 1  2   3   4 |
            |    1   2   3 |
        A = | 2      1   2 |
            |    2       1 |
        LU Decomposition of A gives a diagonal
        banded matrix L and upper triangular matrix U
        '''
        if self.axis==1:
            assert b.ndim > 1
        self._twodia(self.d,self.u1,b,self.axis)
        if self.ndim==2:
            if self.axis==0:
                return self._triangular(self.U, b)
            elif self.axis==1:
                return self._triangular(self.U, b.T)
        else:
            return self._triangular(self.U, b,self.axis)
        
    def set_subsolver(self):
        from scipy.linalg import solve_triangular
        if self.ndim==1:
            # fortrans triangular is faster in 1 dimension
            self._triangular = lafort.triangular.solve_1d
            self._twodia = lafort.tridiagonal.solve_twodia_1d
        elif self.ndim == 2:
            #self._triangular = lafort.triangular.solve_2d
            # scipys solve_triangular is faster in 2 dimensions
            self._triangular = solve_triangular
            self._twodia = lafort.tridiagonal.solve_twodia_2d

    def _lu_decomp(self,A):
        ''' LU Decomposition of A'''
        from scipy.linalg import lu
        P,L,U = lu(A)
        return L,U


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