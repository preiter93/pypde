from .plans import *

class SolverPlan():
    '''
    This class contains all the information to solve
    the linear system of equations.

    You need to add plans that contains information on how to solve
        A x = C b
    This system can be multidimensional. In that case, multiple plans
    are stitched together and executed sequentially. Order matters here.

    Example:
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    from pypde.solver.plans import *
    from pypde.solver.solverplan import *
    A = np.array([[1,1,2],[0,3,6],[0,0,9]])
    b = np.array([[0,1,2],[3,4,6]])
    
    # Make Plans
    rhs = PlanRHS(A,ndim=2,axis=1)
    lhs = PlanLHS(A,ndim=2,axis=1,method="numpy")
    
    # Add them
    solver = SolverPlan()
    solver.add_lhs(lhs)
    solver.add_rhs(rhs)
    
    # Print them
    solver.show_plan()
    
    # Solve them
    rhs = solver.solve_rhs(b)
    x = solver.solve_lhs(rhs)
    print(x)
    
    # The above results should be the same as
    rhs = b@A.T
    np.linalg.solve(A,rhs.T).T
    >  array([[0., 1., 2.],
            [3., 4., 6.]])
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    def __init__(self):
        self.plan_for_lhs = []
        self.plan_for_rhs = []
        self.plan_for_old = []

    def add_rhs(self,plan):
        assert isinstance(plan, PlanRHS)
        self.plan_for_rhs.append(plan)

    def add_old(self,plan):
        assert isinstance(plan, PlanRHS)
        self.plan_for_old.append(plan)

    def add_lhs(self,plan):
        assert isinstance(plan, MetaPlan)
        self.plan_for_lhs.append(plan)

    def solve_rhs(self,b):
        for plan in self.plan_for_rhs:
            b = plan.solve(b)
        return b

    def solve_old(self,b):
        for plan in self.plan_for_old:
            b = plan.solve(b)
        return b

    def solve_lhs(self,b):
        for plan in self.plan_for_lhs:
            b = plan.solve(b)
        return b

    def show_plan(self):
        print("Plans RHS:")
        for i,p in enumerate(self.plan_for_rhs):
            print(i+1,")", self._str_plan(p) )
        print("")
        if self.plan_for_old:
            print("Plans RHS (Old field):")
            for i,p in enumerate(self.plan_for_old):
                print(i+1,")", self._str_plan(p) )
            print("")
        print("Plans LHS:")
        for i,p in enumerate(self.plan_for_lhs):
            print(i+1,")", self._str_plan(p) )
        print("")
        
    def _str_plan(self,plan):
        assert isinstance(plan,MetaPlan)
        return ("Apply method '{:s}' along axis {:1d} ".
        format(plan.flags["method"], plan.flags["axis"]))
