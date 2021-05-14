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

TIME_FFT = 0
TIME_Conv = 0

class NavierStokes(Integrator):
    CONFIG={
        "shape": (50,50),
        "kappa": 1.0,
        "nu": 1.0,
        "dt": 0.2,
        "ndim": 2,
        "tsave": 0.1,
        "dealias": True,
        "integrator": "eu",
        "beta": 1.0,
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

        # Array for rhs
        self.rhs = np.zeros(self.shape)

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

    @property
    @memoized
    def bc_That_cheby(self):
        return galerkin_to_cheby(self.Tbc.vhat,self.Tbc)

    def setup_solver(self):
        from pypde.templates.hholtz import solverplan_hholtz2d_adi
        from pypde.templates.poisson import solverplan_poisson2d

        if self.integrator == "rk3":
            print("Initialize rk3 ...")
            self.set_timestep_coefficients_rk3()
        else:
            print("Initialize euler ...")
            self.set_timestep_coefficients_euler()

        self.solver_U,self.solver_V,self.solver_T = [],[],[]
        for rk in range(self.nstage):
            solver_U = solverplan_hholtz2d_adi(bases=self.U.xs,
            lam=self.dt*self.a[rk]*self.beta*self.nu)
            solver_V = solverplan_hholtz2d_adi(bases=self.V.xs,
            lam=self.dt*self.a[rk]*self.beta*self.nu)
            solver_T = solverplan_hholtz2d_adi(bases=self.T.xs,
            lam=self.dt*self.a[rk]*self.beta*self.kappa)
            self.solver_U.append(solver_U)
            self.solver_V.append(solver_V)
            self.solver_T.append(solver_T)
        self.solver_P = solverplan_poisson2d(self.P.xs,singular=True)

    def set_timestep_coefficients_rk3(self):

        '''
        (1-a_k*L) phi_k = phi_k + b_k*N_k + c_k * N_k-1

        (Diffusion purely implicit)

        RK3:
            a/dt    b/dt    c/dt
            8/15    8/15    0
            2/15    5/12  -17/60
            1/3     3/4    -5/12
        '''
        self.nstage = 3
        self.a = np.array([8./15., 2./15. , 1./3. ])
        self.b = np.array([8./15., 5./12. , 3./4. ])
        self.c = np.array([0     ,-17./60.,-5./12.])

    def set_timestep_coefficients_euler(self):

        '''
        (1-a_k*L) phi_k = phi_k + b_k*N_k + c_k * N_k-1

        (Diffusion purely implicit)
        '''
        self.nstage = 1
        self.a = np.array([1.])
        self.b = np.array([1.])
        self.c = np.array([0 ])

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
        tic = time.perf_counter()
        conv = convective_term(field,ux,uz,
                              deriv_field = self.deriv_field,
                              add_bc = add_bc,
                              dealias= self.dealias )
        global TIME_Conv
        TIME_Conv += time.perf_counter() - tic

        return conv

    def update_U(self,ux,uz,ux_old,uz_old,p,stage):
        tic = time.perf_counter()

        # Pressure term
        dpdx = grad(p,deriv=(1,0),return_field=False)
        rhs = -self.dt*self.a[stage]*dpdx

        # Non-Linear Convection
        rhs -= self.dt*self.b[stage]*self.conv_term(self.U,ux,uz)
        if self.c[stage] != 0:
            rhs -= self.dt*self.c[stage]*self.conv_term(
            self.U,ux_old,uz_old)

        # Add explicit diffusive term
        if self.beta != 1.0:
            rhs += (self.dt*self.a[stage]*(1-self.beta)*self.nu*
            grad(self.U,deriv=(2,0),return_field=False) )
            rhs += (self.dt*self.a[stage]*(1-self.beta)*self.nu*
            grad(self.U,deriv=(0,2),return_field=False) )

        rhs  = self.solver_U[stage].solve_rhs( rhs )
        rhs += self.solver_U[stage].solve_old( self.U.vhat )
        self.U.vhat[:] = self.solver_U[stage].solve_lhs(rhs)

        global TIME_U
        TIME_U += time.perf_counter() - tic

    def update_V(self,ux,uz,ux_old,uz_old,That,p,stage):
        tic = time.perf_counter()

        # Pressure term
        dpdz = grad(p,deriv=(0,1),return_field=False)
        rhs = -self.dt*self.a[stage]*dpdz

        # Non-Linear Convection
        rhs -= self.dt*self.b[stage]*self.conv_term(self.V,ux,uz)
        if self.c[stage] != 0:
            rhs -= self.dt*self.c[stage]*self.conv_term(
            self.V,ux_old,uz_old)

        # Buoyancy
        rhs +=  self.dt*self.a[stage]*That

        # Add explicit diffusive term
        if self.beta != 1.0:
            rhs += (self.dt*self.a[stage]*(1-self.beta)*self.nu*
            grad(self.V,deriv=(2,0),return_field=False) )
            rhs += (self.dt*self.a[stage]*(1-self.beta)*self.nu*
            grad(self.V,deriv=(0,2),return_field=False) )

        rhs  = self.solver_V[stage].solve_rhs( rhs )
        rhs += self.solver_V[stage].solve_old( self.V.vhat )
        self.V.vhat[:] = self.solver_V[stage].solve_lhs(rhs)

        global TIME_V
        TIME_V += time.perf_counter() - tic

    def update_T(self,ux,uz,ux_old,uz_old,stage):
        tic = time.perf_counter()

        # Non-Linear Convection
        rhs = -self.dt*self.b[stage]*self.conv_term(self.T,ux,uz,
        add_bc = uz*self.bc_dTdz)
        if self.c[stage] != 0:
            rhs -= self.dt*self.c[stage]*self.conv_term(
            self.T,ux_old,uz_old,add_bc = uz_old*self.bc_dTdz)

        # Add explicit diffusive term
        if self.beta != 1.0:
            rhs += (self.dt*self.a[stage]*(1-self.beta)*self.kappa*
            grad(self.T,deriv=(2,0),return_field=False) )
            rhs += (self.dt*self.a[stage]*(1-self.beta)*self.kappa*
            grad(self.T,deriv=(0,2),return_field=False) )

        rhs  = self.solver_T[stage].solve_rhs( rhs )
        rhs += self.solver_T[stage].solve_old(self.T.vhat)
        self.T.vhat[:] = self.solver_T[stage].solve_lhs(rhs)

        global TIME_T
        TIME_T += time.perf_counter() - tic

    def update_P(self,fhat,singular=True):
        tic = time.perf_counter()
        rhs  = self.solver_P.solve_rhs(fhat)
        self.P.vhat[:] = self.solver_P.solve_lhs(rhs)
        if singular: self.P.vhat[0,0] = 0

        global TIME_P
        TIME_P += time.perf_counter() - tic

    def update_pres(self,pres,P,div,stage):
        pres.vhat -=  1.0*self.nu*div*self.beta#*self.dt*self.a[stage]#self.dt*self.a[stage]*
        pres.vhat +=  1.0/(self.dt*self.a[stage])*galerkin_to_cheby(
        P.vhat,P)


    def update(self):
        ux_old,uz_old = 0,0
        for rk in range(self.nstage):
            # Buoyancy
            That  = galerkin_to_cheby(self.T.vhat,self.T)
            That += self.bc_That_cheby

            # Convection velocity
            tic = time.perf_counter()
            if self.dealias:
                ux = self.U.dealias.backward(self.U.vhat)
                uz = self.V.dealias.backward(self.V.vhat)
            else:
                ux = self.U.backward(self.U.vhat)
                uz = self.V.backward(self.V.vhat)

            global TIME_FFT
            TIME_FFT += time.perf_counter() - tic

            # Add pressure term
            #self.update_velocity(self.pres,self.U,self.V,
            #fac=self.dt*self.a[rk])

            # Solve Ux
            self.update_U(ux,uz,ux_old,uz_old,self.pres,stage=rk)
            self.update_V(ux,uz,ux_old,uz_old,That,self.pres,stage=rk)

            # Divergence of Velocity
            div = self.divergence_velocity(self.U,self.V)

            # Solve Pressure
            self.update_P(div)

            # Update pressure
            self.update_pres(self.pres,self.P,div,stage=rk)

            # Correct Velocity
            self.update_velocity(self.P,self.U,self.V)

            # Solve Temperature
            self.update_T(ux,uz,ux_old,uz_old,stage=rk)

            ux_old,uz_old = ux,uz

    def callback(self):
        print("Divergence: {:4.2e}".format(
        np.linalg.norm(self.divergence_velocity(self.U,self.V))))


#
# shape = (96,96)
#
# Pr = 1
# Ra = 5e3
# nu = np.sqrt(Pr/Ra)
# kappa = np.sqrt(1/Pr/Ra)
#
# NS = NavierStokes(shape=shape,dt=0.02,tsave=1.0,nu=nu,kappa=kappa,
# dealias=False,integrator="rk3",beta=1.0)
#
# st = time.perf_counter()
# NS.iterate(10.0)
# TIME += time.perf_counter() - st
#
# print("-----------------------")
# print(" Time Convective: {:5.2f} s".format(TIME_Conv))
# print(" Time FFT Conv  : {:5.2f} s".format(TIME_FFT))
# print(" Time Update V  : {:5.2f} s".format(TIME_Update))
# print(" Time Divergence: {:5.2f} s".format(TIME_Divergence))
# print("")
# print(" Time U: {:5.2f} s".format(TIME_U))
# print(" Time V: {:5.2f} s".format(TIME_V))
# print(" Time T: {:5.2f} s".format(TIME_T))
# print(" Time P: {:5.2f} s".format(TIME_P))
# print("")
# print(" Time Total: {:5.2f} s".format(TIME))
# print("-----------------------")
#
# #  Add inhomogeneous part
# for i,v in enumerate(NS.T.V):
#         if NS.T.V[i][0,0] < 0.1: NS.T.V[i] += NS.Tbc.v
#
# anim = NS.T.animate(duration=4,wireframe=False)
# plt.show()
