from pypde import *

def solverplan_poisson1d(bases,singular=False):
    '''
    Poisson equation:
        D2 u = rhs
    Premultiplied with pseudoinverse of D2:
        I S*v = B*rhs

    Input
        base: list of MetaBases
            ChebDirichlet ("CD") or Chebneumann ("CN")
        singular: bool (default = False)
            For pure neumann cases, add a small value
            to the zero eigenvalue
            Set constant part after calculation to
            zero by hand. See "How to" in solverplan_poisson2d

    Returns
        Solverplan
    '''
    field = Field(bases)
    assert field.ndim==1
    # --- Matrices ----
    Sx = field.xs[0].S_sp
    Bx = field.xs[0].family.B(2,2)
    Ix = field.xs[0].family.I(2)
    Ax = Ix@Sx
    if singular:
        assert  Ax[0,0] == 0, "Matrix does not look singular"
        # Add very small term to make system non-singular
        Ax[0,0] += 1e-20

    # --- Solver Plans ---
    solver = SolverPlan()
    solver.add_rhs(PlanRHS(Bx, ndim=1,axis=0))    # rhs
    solver.add_lhs(PlanLHS(Ax, ndim=1,axis=0,method="twodma") ) #lhs

    return solver

def solverplan_poisson2d(bases,singular=False):
    '''
    Poisson equation:
        D2 u = rhs
    Premultiplied with pseudoinverse of D2:
        I S*v = B*rhs

    Bases must be pure chebyshev

    This plan uses eigendecomposition in the second
    dimension to separate the dimensions

    Input
        base: list of MetaBases
            ChebDirichlet ("CD") or Chebneumann ("CN")
        singular: bool (default = False)
            For pure neumann cases, add a small value
            to the zero eigenvalue
            Set constant part after calculation to
            zero by hand. See "How to"

    Returns
        Solverplan

    "How to" use solverplan
    rhs  = self.solver.solve_rhs(rhs)
    vhat[:] = self.solver.solve_lhs(rhs)
    # if singular, set constant part to zero
    # vhat[0] = 0
    '''
    from pypde.solver.utils import eigdecomp

    field = Field(bases)
    assert field.ndim==2
    # --- Matrices ----
    Sx = field.xs[0].S_sp
    Bx = field.xs[0].family.B(2,2)
    Ix = field.xs[0].family.I(2)
    Ax = Ix@Sx
    Cx = Bx@Sx

    Sy = field.xs[1].S_sp
    By = field.xs[1].family.B(2,2)
    Iy = field.xs[1].family.I(2)
    Ay = Iy@Sy
    Cy = By@Sy

    # -- Eigendecomposition ---
    CyI = np.linalg.inv(Cy)
    wy,Qy,QyI = eigdecomp( CyI@Ay )
    if singular:
        wy[0] += 1e-20

    Hy = QyI@CyI@By

    # --- Solver Plans ---
    solver = SolverPlan()
    solver.add_rhs(PlanRHS(Bx,ndim=2,axis=0))
    solver.add_rhs(PlanRHS(Hy,ndim=2,axis=1))

    solver.add_lhs( PlanLHS(Ax,alpha=wy,C=Cx,ndim=2,axis=0,
    method="poisson",singular=True) )
    solver.add_lhs( PlanLHS(Qy,ndim=2,axis=1,method="multiply") )

    return solver
