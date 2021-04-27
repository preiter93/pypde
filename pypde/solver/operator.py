from .matrix import MatrixRHS, MatrixLHS

class RHSExplicit():
    '''
    Class handles rhs (b) and constant rhs (f) of the system of equations:
        Ax = b + f
    or
        Ax = PM@(b +f)
    when premultiply is specified
        
    Input:
        b: ndarray (ndim)
            Add rhs part b. More b's can be added by self.add(b).
        f: ndarray (ndim)
            Does not change from iteration to iteration (static)
        premultiply: MatrixBase (optional) 
            Premultiply rhs at each step with sparse Matrix or scalar.
            More can be added via self.add_PM(PM).
    
    Use:
        >>> b = np.array([1,2,3])
        >>> P = np.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> PM = MatrixRHS(P,axis=0)
        >>> R = RHSExplicit(b)
        >>> R.add_PM(PM)
        >>> R.rhs
        >>> array([14, 32, 50]) # = P@b
    '''
    def __init__(self,b=None,premultiply=None,f=None):
       	self._b = 0 if b is None else b
        self._f = 0 if f is None else f
        self.PM = []
        if premultiply is not None:
            self.add_PM(premultiply)
            
    @property    
    def f(self):
        return self._f

    @property    
    def b(self):
        return self._b
    
    @b.setter
    def b(self,value):
        self._b = value
        
    def add(self, b):
        self.b += b
    
    def add_PM(self,PM):
        assert isinstance(PM,MatrixRHS), "PM must be instance Matrixbase"
        assert PM.sparse, "PM must be sparse"
        self.PM.append(PM)
    
    @property
    def rhs(self):
        if self.PM:
            x = self.b + self.f
            for P in self.PM:
                x = P.dot(x)
            return x 
        return self.b+self.f

class LHSImplicit():
    '''
    Class handles lhs (A) of the system of equations:
        Ax = b
    and defines which strategy should be used to solve
    the system of equations.
    
    Input
        A: MatrixLHS (optional) 
            Contains lhs matrices and information about the solver strategy.
            Additional lhs' can be added using LHSImplicit.add(A)
    
    Use
        >>> A = np.array([[1,2,3],[0,4,5],[0,0,7]])
        >>> b = np.array([[10,10,10],[11,11,11],[12,12,12]])
        >>> MA = MatrixLHS(A,ndim=2,axis=0,solver="uptria")
        >>> L = LHSImplicit(MA)
        >>> L.solve(b)
        >>> assert np.allclose(x,np.linalg.solve(A,b))
    '''
    
    def __init__(self,A=None):
        self.A = []
        if A is not None:
            self.add(A)
        
    def add(self,A):
        assert isinstance(A,MatrixLHS), "PM must be instance MatrixLHS"
        self.A.append(A)
        
    def solve(self,b,*args,**kwargs):
        if self.A:
            for a in self.A:
                b = a.solve(b,*args,**kwargs)
        return b