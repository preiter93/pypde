from pypde import *

def solverplan_hholtz1d(shape,base,lam):
    '''
    Helmholtz equation:
        (1 - lam*D2) u = rhs + uold
    Premultiplied with pseudoinverse of D2:
        (B - lam*I) S*v = B*rhs + B*S*vold

    Input
        shape: 1 tuple int
            Shape of domain in physical space
        base: 1 tuple str
            ChebDirichlet ("CD") or Chebneumann ("CN")
        lam: float
            See equation above

    Returns
        Solverplan

    How to use solverplan
    rhs  = self.solver.solve_rhs(rhs)
    rhs += self.solver.solve_old(vold)
    vhat[:] = self.solver.solve_lhs(rhs)
    '''
    field = Field(shape,base)
    assert field.ndim==1
    # --- Matrices ----
    Sx = field.xs[0].S_sp
    Bx = field.xs[0].family.B(2,2)
    Ix = field.xs[0].family.I(2)
    Ax = Bx@Sx-lam*Ix@Sx

    # --- Solver Plans ---
    solver = SolverPlan()
    solver.add_rhs(PlanRHS(Bx,   ndim=1,axis=0))    # rhs
    solver.add_old(PlanRHS(Bx@Sx,ndim=1,axis=0))    # rhs (uold)
    solver.add_lhs(PlanLHS(Ax,   ndim=1,axis=0,method="fdma") ) #lhs

    return solver

def solverplan_hholtz2d_adi(shape,base,lam):
    '''
    Helmholtz equation:
        (1 - lam*D2) u = rhs + uold
    Premultiplied with pseudoinverse of D2:
        (B - lam*I) S*v = B*rhs + B*S*vold

    Bases must be pure chebyshev

    This plan uses the alternating direction method
    (ADI) to separate the dimensions

    Input
        shape: 2 tuple int
            Shape of domain in physical space
        base: 2 tuple str
            ChebDirichlet ("CD") or Chebneumann ("CN")
        lam: float
            See equation above

    Returns
        Solverplan

    How to use solverplan
    rhs  = self.solver.solve_rhs(rhs)
    rhs += self.solver.solve_old(vold)
    vhat[:] = self.solver.solve_lhs(rhs)
    '''
    field = Field(shape,base)
    assert field.ndim==2
    # --- Matrices ----
    Sx = field.xs[0].S_sp
    Bx = field.xs[0].family.B(2,2)
    Ix = field.xs[0].family.I(2)
    Ax = Bx@Sx-lam*Ix@Sx

    Sy = field.xs[1].S_sp
    By = field.xs[1].family.B(2,2)
    Iy = field.xs[1].family.I(2)
    Ay = By@Sy-lam*Iy@Sy

    # --- Solver Plans ---
    solver = SolverPlan()
    solver.add_rhs(PlanRHS(Bx,   ndim=2,axis=0))    # rhs
    solver.add_old(PlanRHS(Bx@Sx,ndim=2,axis=0))    # rhs (uold)
    solver.add_lhs(PlanLHS(Ax,   ndim=2,axis=0,method="fdma") ) #lhs

    solver.add_rhs(PlanRHS(By,   ndim=2,axis=1))    # rhs
    solver.add_old(PlanRHS(By@Sy,ndim=2,axis=1))    # rhs (uold)
    solver.add_lhs(PlanLHS(Ay,   ndim=2,axis=1,method="fdma") ) #lhs

    return solver