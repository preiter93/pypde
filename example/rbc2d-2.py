import sys
sys.path.append("./")
from pypde import *
from pypde.plot import *
import matplotlib.pyplot as plt
import time
initplot()

TIME = 0
TIME_U = 0
TIME_V = 0
TIME_P = 0
TIME_T = 0
TIME_Update = 0
TIME_Divergence = 0

class NavierStokes(Integrator):
    CONFIG={
        "shape": (50,50),
        "kappa": 1.0,
        "nu": 1.0,
        "dt": 0.2,
        "ndim": 2,
        "tsave": 0.1,
        "dealias": True,
    }
    def __init__(self,**kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)

        # Space for Fields
        self.T = Field( [Base(self.shape[0],"CN"),Base(self.shape[1],"CD")] )
        self.U = Field( [Base(self.shape[0],"CD",dealias=3/2),Base(self.shape[1],"CD",dealias=3/2)] )
        self.V = Field( [Base(self.shape[0],"CD",dealias=3/2),Base(self.shape[1],"CD",dealias=3/2)] )
        self.P = Field( [Base(self.shape[0],"CN"),Base(self.shape[1],"CN")] )
        # Space for derivatives
        self.deriv_field = Field( [Base(self.shape[0],"CH",dealias=3/2),Base(self.shape[1],"CH",dealias=3/2)] )
        # Additional pressure
        self.pres = Field( [Base(self.shape[0],"CH"),Base(self.shape[1],"CH")] )

        # Setup Solver solverplans
        self.setup_solver()

        # Coordinates
        self.x,self.y = self.T.x, self.T.y
        self.xx,self.yy = np.meshgrid(self.x,self.y,indexing="ij")

        # Setup Temperature field and bcs
        self.set_temperature()
        self.set_temp_fieldbc()

    def set_temperature(self):
        self.T.v = np.sin(0.5*np.pi*self.xx)*np.cos(0.5*np.pi*self.yy)
        #self.T.v[:] = 0
        self.T.forward()

    def set_temp_fieldbc(self):
        ''' Setup Inhomogeneous field for temperature'''
        # Boundary Conditions T (y=-1; y=1)
        bc = np.zeros((self.shape[0],2))
        bc[:,0],bc[:,1] =  0.5,-0.5
        self.Tbc = FieldBC(self.T.xs,axis=1)
        self.Tbc.add_bc(bc)

    @property
    @memoized
    def bc_d2Tdz2(self):
        return grad(self.Tbc,deriv=(0,2),return_field=False)

    @property
    @memoized
    def bc_dTdz(self):
        vhat =  grad(self.Tbc,deriv=(0,1),return_field=False)
        if self.dealias:
            return self.deriv_field.dealias.backward(vhat)
        return self.deriv_field.backward(vhat)

    def setup_solver(self):
        from pypde.templates.hholtz import solverplan_hholtz2d_adi
        from pypde.templates.poisson import solverplan_poisson2d
        self.solver_U = solverplan_hholtz2d_adi(
            bases=self.U.xs,lam=self.dt*self.nu)
        self.solver_V = solverplan_hholtz2d_adi(
            bases=self.V.xs,lam=self.dt*self.nu)
        self.solver_T = solverplan_hholtz2d_adi(
            bases=self.T.xs,lam=self.dt*self.kappa)
        self.solver_P = solverplan_poisson2d(self.P.xs,
            singular=True)

    def update_time(self):
        self.T.t += self.dt
        self.P.t += self.dt
        self.U.t += self.dt
        self.V.t += self.dt
        self.time += self.dt

    def save(self):
        self.T.save()
        self.P.save()
        self.U.save()
        self.V.save()

    def update_velocity(self,p,u,v,fac=1.0):
        tic = time.perf_counter()

        dpdx = grad(p,deriv=(1,0),return_field=False)
        dpdz = grad(p,deriv=(0,1),return_field=False)

        u.vhat -= cheby_to_galerkin(dpdx*fac,u)
        v.vhat -= cheby_to_galerkin(dpdz*fac,v)

        test = cheby_to_galerkin(dpdx,u)
        test = cheby_to_galerkin(dpdx,u)

        global TIME_Update
        TIME_Update += time.perf_counter() - tic


    def divergence_velocity(self,u,v):
        tic = time.perf_counter()

        dudx = grad(u,deriv=(1,0),return_field=False)
        dudz = grad(v,deriv=(0,1),return_field=False)

        global TIME_Divergence
        TIME_Divergence += time.perf_counter() - tic

        return dudx + dudz

    def conv_term(self,field,ux,uz,add_bc=None):
        conv = convective_term(field,ux,uz,
                              deriv_field = self.deriv_field,
                              add_bc = add_bc,
                              dealias= self.dealias )
        return conv

    def update_U(self,ux,uz):
        tic = time.perf_counter()
        rhs  = -self.dt*self.conv_term(self.U,ux,uz)
        rhs  = self.solver_U.solve_rhs( rhs )
        rhs += self.solver_U.solve_old( self.U.vhat )
        self.U.vhat[:] = self.solver_U.solve_lhs(rhs)

        global TIME_U
        TIME_U += time.perf_counter() - tic

    def update_V(self,ux,uz,That):
        tic = time.perf_counter()
        rhs  = -self.dt*self.conv_term(self.V,ux,uz)
        rhs +=  self.dt*That
        rhs  = self.solver_V.solve_rhs( rhs )
        rhs += self.solver_V.solve_old( self.V.vhat )
        self.V.vhat[:] = self.solver_V.solve_lhs(rhs)

        global TIME_V
        TIME_V += time.perf_counter() - tic

    def update_T(self,ux,uz):
        tic = time.perf_counter()
        rhs  = self.dt*self.kappa*self.bc_d2Tdz2
        rhs -= self.dt*self.conv_term(self.T,ux,uz,
        add_bc = uz*self.bc_dTdz)
        rhs  = self.solver_T.solve_rhs( rhs )
        rhs += self.solver_T.solve_old(self.T.vhat)
        self.T.vhat[:] = self.solver_T.solve_lhs(rhs)

        global TIME_T
        TIME_T += time.perf_counter() - tic

    def update_P(self,fhat,singular=True):
        tic = time.perf_counter()
        rhs  = self.solver_P.solve_rhs(fhat)
        self.P.vhat[:] = self.solver_P.solve_lhs(rhs)
        if singular: self.P.vhat[0,0] = 0

        global TIME_P
        TIME_P += time.perf_counter() - tic

    def update(self):
        # Buoyancy
        That  = galerkin_to_cheby(self.T.vhat,self.T)
        That += galerkin_to_cheby(self.Tbc.vhat,self.Tbc)

        # Convection velocity
        if self.dealias:
            ux = self.U.dealias.backward(self.U.vhat)
            uz = self.V.dealias.backward(self.V.vhat)
        else:
            ux = self.U.backward(self.U.vhat)
            uz = self.V.backward(self.V.vhat)

        # Add pressure term
        self.update_velocity(self.pres,self.U,self.V,fac=self.dt)

        # Solve Ux
        self.update_U(ux,uz)
        self.update_V(ux,uz,That)

        # Divergence of Velocity
        div = self.divergence_velocity(self.U,self.V)#(dudx + dudz)

        # Solve Pressure
        self.update_P(div)

        # Update pressure
        self.pres.vhat += -1.0*self.dt*self.nu*div
        self.pres.vhat +=  1.0/self.dt*galerkin_to_cheby(self.P.vhat,self.P)

        # Correct Velocity
        self.update_velocity(self.P,self.U,self.V)

        #f = dudx[5:-5,5:-5] + dudz[5:-5,5:-5]
        #print("Divergence: {:4.2e}".format(np.linalg.norm(f)))

        # Solve Temperature
        self.update_T(ux,uz)



shape = (96,96)

Pr = 1
Ra = 5e4
nu = np.sqrt(Pr/Ra)
kappa = np.sqrt(1/Pr/Ra)

NS = NavierStokes(shape=shape,dt=0.02,tsave=2.0,nu=nu,kappa=kappa,dealias=False)

st = time.perf_counter()
NS.iterate(20.0)
TIME += time.perf_counter() - st

print("-----------------------")
print(" Time Update V  : {:5.2f} s".format(TIME_Update))
print(" Time Divergence: {:5.2f} s".format(TIME_Divergence))
print("")
print(" Time U: {:5.2f} s".format(TIME_U))
print(" Time V: {:5.2f} s".format(TIME_V))
print(" Time T: {:5.2f} s".format(TIME_T))
print(" Time P: {:5.2f} s".format(TIME_P))
print("")
print(" Time Total: {:5.2f} s".format(TIME))
print("-----------------------")

#  Add inhomogeneous part
for i,v in enumerate(NS.T.V):
        if NS.T.V[i][0,0] < 0.1: NS.T.V[i] += NS.Tbc.v

anim = NS.T.animate(duration=4,wireframe=False)
plt.show()
