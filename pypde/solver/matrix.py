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
        self._check_input(A)
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

    def _check_input(self,A):
        assert np.any([isinstance(A,float),
            isinstance(A,int),
            sp.issparse(A),
            isinstance(A,np.ndarray)]), \
        "MatrixBase: Input type does not match any method!"
    
class MatrixRHS(MatrixBase):
    def __init__(self,A,axis=0):
        MatrixBase.__init__(self,A,axis,sparse=True)  

from pypde.solver.fortran import linalg as lafort

class MatrixLHS(MatrixBase):
    '''
    Extends MatrixBase and adds functionality to solve
        (lam*A +C) x = b
    where the type of solver should be chosen by the 
    type of lhs matrix.

    Note that C and lambda are optional
    
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
            
            'matmul': Simple matrix multiplication x = A@b, used in 
                      2-dimensional domains
            
            'poisson': Solve 2D Poisson problems: 
                Note: A helper must be supplied containing the eigenvalues
                      lambda i.e. init(.., helper=lambda)
    '''
    all_methods = [
        "solve",
        "tdma",
        "fdma",
        "matmul",
        "poisson"
    ]

    CONFIG = {
        "transpose": True, # Transpose Matrix for operations on axis 1
        "skip": 0,         # Skip parts (Neumann singularity) Not implemented
    }

    def __init__(self,A,ndim,axis,C=None,lam=None,sparse=False,solver="solve",**kwargs):
        assert solver in self.all_methods, "Solver type not found!"
        MatrixBase.__init__(self,A,axis,sparse=sparse)
        self.C = C
        self.lam = lam

        self.__dict__.update(self.CONFIG)
        self.__dict__.update(kwargs)

        self.solver = solver
        self.ndim = ndim
        self.set_subsolver()

        if solver in ["solve"]:
            self.solve = self.solver_solve

        if solver in ["tdma"]:
            L = self.A
            self.d, self.u1 = np.diag(L).copy(),np.diag(L,+2).copy()
            self.solve = self.solve_tdma if self.ndim==1 else self.solve_tdma2d

        if solver in ["fdma"]:
            self.l,self.d,self.u1,self.u2 = self._init_fdma(self.A,True)
            self.solve = self.solve_fdma if self.ndim==1 else self.solve_fdma2d

        if solver in ["matmul"]:
            self.solve = self.matmul

        if solver in ["poisson"]:
            assert self.lam is not None or self.C is not None, \
            "Solver type 'poisson' needs lam array (lam=eigenvalues) and C matrix!"
            assert self.lam.ndim==1,  "Eigenvalues must be 1D array!"
            assert self.C.ndim==2,     "C Matrix must be 2D matrix!"
            assert self.ndim == 2 
            self.solve = self.solve_poisson

    def _init_fdma(self,A,fortran=True):
        ''' 
        Initialize fdma solver 
        '''
        if fortran:
            d,u1,u2,l = lafort.tridiagonal.init_fdma(A,A.shape[0])
        else:
            d, u1 = np.diag(A).copy(), np.diag(A,+2).copy()
            u2, l = np.diag(A,+4).copy(), np.diag(A,-2).copy()
            FDMA_LU(l, d, u1, u2)
        return l,d,u1,u2

    def solver_solve(self,b):
        ''' 
        Solve Ax=b, where A is of any kind
        '''
        assert self.ndim == b.ndim, "Dimensionality mismatch in MatrixLHS. Check ndim."
        if self.axis==0:
            return np.linalg.solve(self.A,b)
        elif self.axis==1:
            assert b.ndim > 1
            return np.linalg.solve(self.A,b.T).T

    def solve_tdma(self,b):
        '''
        Solve Ax = b, where A is
        2-diagonal matrix with diagonals in offsets 0, 2
        Input
            b: 1d array
        '''
        self._tdma(self.d,self.u1,b)
        return b

    def solve_fdma(self,b):
        '''
        Solve Ax = b, where A is
        4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
        Input
            b: 1d array
        '''
        self._fdma(self.d,self.u1,self.u2,self.l,b)
        return b

    def solve_tdma2d(self,b):
        '''
         Solve Ax = b, where A is
         2-diagonal matrix with diagonals in offsets 0, 2
         Input
            b: 2d array
        '''
        self._tdma(self.d,self.u1,b,self.axis)
        return b

    def solve_fdma2d(self,b):
        '''
        Solve Ax = b, where A is
        4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
        Input
            b: 2d array
        '''
        self._fdma(self.d,self.u1,self.u2,self.l,b,self.axis)
        return b


    def solve_poisson(self,b):
        '''
        Following type of equation arises in 2D Poisson problems: 
                    
                (M*diag(lam_i) + D) u_i = f_i
        where
            M: matrix with diagonals in offsets -2, 0, 2, 4
            D: matrix with diagonals in offsets  0, 2
            lam: array of size b.shape[1] (b.shape[0]) if axis=0 (axis=1) 
        '''
        _fdma2 = lafort.tridiagonal.solve_fdma2d_type2
        m = b.shape[1] if self.axis==0 else b.shape[0]
        assert self.lam.size == m, \
        "Size of eigenvalue array does not match!"

        _fdma2(self.A,self.C,self.lam,b,self.axis)
            
        return b

    def matmul(self,b):
        '''
        Matrixmultiplication "Projection"
            x = M@b   (axis = 0)
            x = b@M.T (axis = 1)
        '''
        assert b.ndim==2, "Input must be a 2dimensional matrix"
        if self.axis==0:
            return self.A@b
        return b@self.A.T if self.transpose else b@self.A


    def set_subsolver(self):
        if self.ndim==1:
            self._tdma = lafort.tridiagonal.solve_tdma
            self._fdma = lafort.tridiagonal.solve_fdma
        if self.ndim==2:
            self._tdma = lafort.tridiagonal.solve_tdma2d
            self._fdma = lafort.tridiagonal.solve_fdma2d

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


    
