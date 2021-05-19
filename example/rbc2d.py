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
        self.T.v = np.sin(0.5*np.pi*self.xx)*np.cos(0.5*np.pi*self.yy)*0.5
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


    # --- Post processing ----

    def callback(self):
        print("Divergence: {:4.2e}".format(
        np.linalg.norm(self.divergence_velocity(self.U,self.V))))

    def plot(self,skip=None,return_fig=False):
         #-- Plot 
        self.T.backward(); self.U.backward(); self.V.backward()

        fig,ax = plt.subplots()
        ax.contourf(self.xx,self.yy,self.T.v+self.Tbc.v, 
            levels=np.linspace(-0.5,0.5,40))
        ax.set_aspect(1)

        # Quiver
        speed = np.max(np.sqrt(self.U.v**2+self.V.v**2))
        if skip is None:
            skip = self.shape[0]//16
        ax.quiver(self.xx[::skip,::skip],self.yy[::skip,::skip],
                self.U.v[::skip,::skip]/speed,self.V.v[::skip,::skip]/speed,
                scale=7.9,width=0.007,alpha=0.5,headwidth=4)
        if return_fig:
            return plt,ax
        plt.show()

    def eval_Nu(self):
        from pypde.field_operations import eval_Nu,eval_Nuvol
        Nuz = eval_Nu(self.T,self.deriv_field)
        Nuv = eval_Nuvol(self.T,self.V,self.kappa,self.deriv_field,Tbc=self.Tbc)
        return Nuz,Nuv

    def interpolate(self,NS_old,spectral=True):
        from pypde.field_operations import interpolate
        interpolate(NS_old.T,self.T,spectral)
        interpolate(NS_old.U,self.U,spectral)
        interpolate(NS_old.V,self.V,spectral) 

    def write(self,leading_str="",add_time=True):
        Tname = leading_str + "T"
        Uname = leading_str + "U"
        Vname = leading_str + "V"
        self.T.backward()
        self.U.backward()
        self.V.backward()
        dict = {"nu": self.nu,"kappa": self.kappa}
        self.T.write(filename=None,leading_str=Tname,add_time=add_time,dict=dict)
        self.U.write(filename=None,leading_str=Uname,add_time=add_time,dict=dict)
        self.V.write(filename=None,leading_str=Vname,add_time=add_time,dict=dict)

    def read(self,leading_str="",add_time=True):
        Tname = leading_str + "T"
        Uname = leading_str + "U"
        Vname = leading_str + "V"
        dict = {"nu": self.nu,"kappa": self.kappa}
        self.T.read(filename=None,leading_str=Tname,add_time=add_time,dict=dict)
        self.U.read(filename=None,leading_str=Uname,add_time=add_time,dict=dict)
        self.V.read(filename=None,leading_str=Vname,add_time=add_time,dict=dict)

    # --- For steady state calculations ----
    def flatten(self):
        return (self.T.vhat.flatten().copy(), 
            self.U.vhat.flatten().copy(), 
            self.V.vhat.flatten().copy())

    def reshape(self,X):
        T_mask,U_mask,V_mask = self.get_masks()
        That = X[T_mask].copy().reshape(self.T.vhat.shape)
        Uhat = X[U_mask].copy().reshape(self.U.vhat.shape)
        Vhat = X[V_mask].copy().reshape(self.V.vhat.shape)
        return That,Uhat,Vhat 

    def vectorify(self):
        return np.concatenate((self.flatten()))

    def get_masks(self):
        t,u,v = self.flatten()
        T_mask = slice(0,t.size)
        U_mask = slice(t.size,t.size+u.size)
        V_mask = slice(t.size+u.size,t.size+u.size+v.size)
        return T_mask, U_mask, V_mask

    def steady_fun(X, NS):
        '''
        Input:
            X: ndarray (1D)
                Flow field vector [T,u,v]
                
        Output
            ndarry (1D)
                Residual vector [Tr,ur,v]
        '''
        NS.T.vhat[:],NS.U.vhat[:],NS.V.vhat[:] = NS.reshape(X)
        NS.update()
        Y = NS.vectorify()
        return (Y-X)/NS.dt

    def solve_steady_state(self,X0=None,maxiter=300,disp=True,tol=1e-4):
        '''
        Solve steady state using scipy's GMRES algorithm
        '''
        from scipy import optimize
        ''' Solve steady state '''
        options={"maxiter": maxiter, "disp": disp, "fatol": tol}
        if X0 is None:
            X0 = self.vectorify()
        sol = optimize.root(steady_fun, X0, args=(self,), method="krylov",options=options)
        return sol



    


'''
Example for steady state calculations 

import numpy as np 
import matplotlib.pyplot as plt
from pypde import *
from example.rbc2d import NavierStokes,nu,kappa,steady_fun
from pypde.field_operations import eval_Nu, eval_Nuvol
from scipy import optimize

# -- Initialize Navier Stokes Solver
shape = (128,128)
Ra = 5e3
Pr = 1
NS = NavierStokes(shape=shape,dt=0.1,tsave=20.,
                  nu=nu(Ra/2.**3,Pr),kappa=kappa(Ra/2.**3,Pr),
                  dealias=True,integrator="eu",beta=1.0)
NS.iterate(40) # iterate first to find good initial guess
NS.plot()

# -- Solve Steady State
# Initial guess
X0 = NS.vectorify()

options={"maxiter": 300, "disp": True, "fatol": 1e-4}
sol = optimize.root(steady_fun, X0, args=(NS,), method="krylov",options=options)
X0 = sol.x
'''
def steady_fun(X, NS):
    '''
    Input:
        X: ndarray (1D)
            Flow field vector [T,u,v]
            
    Output
        ndarry (1D)
            Residual vector [Tr,ur,v]
    # '''
    NS.T.vhat[:],NS.U.vhat[:],NS.V.vhat[:] = NS.reshape(X)
    NS.update()
    Y = NS.vectorify()
    return (Y-X)/NS.dt


def fun(X):
    '''
    Input:
        X: ndarray (1D)
            Flow field vector [T,u,v]
            
    Output
        ndarry (1D)
            Residual vector [Tr,ur,v]
    # '''
    # T = X[T_mask].copy()
    # U = X[U_mask].copy()
    # V = X[V_mask].copy()
    # NS.T.vhat = T.reshape(NS.T.vhat.shape).copy()
    # NS.U.vhat = U.reshape(NS.U.vhat.shape).copy()
    # NS.V.vhat = V.reshape(NS.V.vhat.shape).copy()
    NS.T.vhat[:],NS.U.vhat[:],NS.V.vhat[:] = reshape(X,NS)
    
    NS.update()
    
    # t,u,v = NS.T.vhat.flatten().copy(), NS.U.vhat.flatten().copy(), NS.V.vhat.flatten().copy()
    # Y = np.concatenate( (t,u,v) )
    Y = vectorify(NS)
    return (Y-X)


def flatten(NS):
    return (NS.T.vhat.flatten().copy(), 
        NS.U.vhat.flatten().copy(), 
        NS.V.vhat.flatten().copy()
        )

def reshape(X,NS):
    T_mask,U_mask,V_mask = get_masks(NS)
    That = X[T_mask].copy().reshape(NS.T.vhat.shape)
    Uhat = X[U_mask].copy().reshape(NS.U.vhat.shape)
    Vhat = X[V_mask].copy().reshape(NS.V.vhat.shape)
    return That,Uhat,Vhat 

def vectorify(NS):
    return np.concatenate((flatten(NS)))

def get_masks(NS):
    t,u,v = flatten(NS.T.vhat,NS.U.vhat,NS.V.vhat)
    T_mask = slice(0,t.size)
    U_mask = slice(t.size,t.size+u.size)
    V_mask = slice(t.size+u.size,t.size+u.size+v.size)
    return T_mask, U_mask, V_mask


def nu(Ra,Pr):
    return np.sqrt(Pr/Ra)

def kappa(Ra,Pr):
    return  np.sqrt(1/Pr/Ra)

