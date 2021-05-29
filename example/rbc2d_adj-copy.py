import sys
sys.path.append("./")
from pypde import *
from pypde.plot import *
import matplotlib.pyplot as plt
import time
from rbc2d import NavierStokes,nu,kappa

def conv_term(v_field, u, deriv, deriv_field=None, dealias=False):
    '''
    Calculate
        u*dvdx

    Input
        v_field: class Field
            Contains field variable vhat in spectral space
        ux,uz:  ndarray
            (Dealiased) velocity fields in physical space
        deriv: tuple
            (1,0) for partial_x, (0,1) for partial_z
        deriv_field: field (optional)
            Field (space) where derivatives life
        dealias: bool (optional)
            Dealias convective term. In this case, input ux and
            uz must already be dealiased and deriv_field must
            be initialized with ,dealias=3/2

    Return
        Field of (dealiased) convective term in physical space
        Transform to spectral space via conv_field.forward()
    '''
    assert isinstance(v_field, Field), "v_field must be instance Field"

    if deriv_field is None:
        if dealias:
            deriv_field = Field( [
            Base(vx_field.shape[0],"CH",dealias=3/2),
            Base(vx_field.shape[1],"CH",dealias=3/2)] )
        else:
            deriv_field = Field( [
            Base(vx_field.shape[0],"CH"),
            Base(vx_field.shape[1],"CH")] )

    # dvdx
    vhat = grad(v_field,deriv,return_field=False)
    if dealias:
        dvdx = deriv_field.dealias.backward(vhat)
    else:
        dvdx = deriv_field.backward(vhat)
    conv = dvdx*u

    if dealias:
        return deriv_field.dealias.forward(conv)
    return deriv_field.forward(conv)

class NavierStokesAdjoint(Integrator):
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

        # Initialize underlying NavierStokes
        self.NS = NavierStokes(**self.__dict__)

        # Space for Fields
        self.T = Field( [Base(self.shape[0],"CN",dealias=3/2),Base(self.shape[1],"CD",dealias=3/2)] )
        self.U = Field( [Base(self.shape[0],"CD",dealias=3/2),Base(self.shape[1],"CD",dealias=3/2)] )
        self.V = Field( [Base(self.shape[0],"CD",dealias=3/2),Base(self.shape[1],"CD",dealias=3/2)] )
        self.P = Field( [Base(self.shape[0],"CN"),Base(self.shape[1],"CN")] )

        # Space for Adjoint Fields
        self.TA = Field( [Base(self.shape[0],"CN",dealias=3/2),Base(self.shape[1],"CD",dealias=3/2)] )
        self.UA = Field( [Base(self.shape[0],"CD",dealias=3/2),Base(self.shape[1],"CD",dealias=3/2)] )
        self.VA = Field( [Base(self.shape[0],"CD",dealias=3/2),Base(self.shape[1],"CD",dealias=3/2)] )

        # Space for derivatives
        self.deriv_field = Field( [Base(self.shape[0],"CH",dealias=3/2),Base(self.shape[1],"CH",dealias=3/2)] )

        # Additional pressure field
        self.pres = Field( [Base(self.shape[0],"CH"),Base(self.shape[1],"CH")] )

        # Setup Solver solverplans
        self.setup_solver()

        # Coordinates
        self.x,self.y = self.T.x, self.T.y
        self.xx,self.yy = np.meshgrid(self.x,self.y,indexing="ij")

        # Setup Temperature field and bcs
        #self.set_temperature()

        # Array for rhs's
        self.rhs = np.zeros(self.shape)

        # Temperature BC in physical space
        self.deriv_field.vhat[:] = galerkin_to_cheby(self.NS.Tbc.vhat, self.NS.Tbc)
        if self.dealias:
            self.temp_bc = self.deriv_field.dealias.backward(self.deriv_field.vhat)
        else:
            self.temp_bc = self.deriv_field.backward(self.deriv_field.vhat)

    def set_temperature(self,amplitude=0.5):
        self.T.v = amplitude*np.sin(0.5*np.pi*self.xx)*np.cos(0.5*np.pi*self.yy)
        self.T.forward()

    def setup_solver(self):
        from pypde.templates.poisson import solverplan_poisson2d

        # Time step coefficients
        self.a = self.NS.a
        self.b = self.NS.b
        self.c = self.NS.c
        self.nstage = self.NS.nstage

        # Solver Plans
        self.solver_P = self.NS.solver_P

        # Plans to smooth fields
        self.nabla_U = solverplan_poisson2d(self.U.xs,singular=False)
        self.nabla_V = solverplan_poisson2d(self.V.xs,singular=False)
        self.nabla_T = solverplan_poisson2d(self.T.xs,singular=False)

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

    def plot(self,skip=None,return_fig=False):
         #-- Plot
        self.T.backward(); self.U.backward(); self.V.backward()

        fig,ax = plt.subplots()
        ax.contourf(self.xx,self.yy,self.T.v+self.NS.Tbc.v,
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

    def conv_term_adj_ux(self,fieldx,fieldz,fieldT,ux,uz,temp,add_bc=None):
        conv  = conv_term(fieldx,ux,deriv=(1,0),
                              deriv_field = self.deriv_field,
                              dealias= self.dealias )

        conv += conv_term(fieldz,uz,deriv=(1,0),
                              deriv_field = self.deriv_field,
                              dealias= self.dealias )

        conv += conv_term(fieldT,temp,deriv=(1,0),
                              deriv_field = self.deriv_field,
                              dealias= self.dealias )

        conv += conv_term(fieldT,self.temp_bc,deriv=(1,0),
                              deriv_field = self.deriv_field,
                              dealias= self.dealias )

        return conv

    def conv_term_adj_uz(self,fieldx,fieldz,fieldT,ux,uz,temp,add_bc=None):
        conv  = conv_term(fieldx,ux,deriv=(0,1),
                              deriv_field = self.deriv_field,
                              dealias= self.dealias )

        conv += conv_term(fieldz,uz,deriv=(0,1),
                              deriv_field = self.deriv_field,
                              dealias= self.dealias )

        conv += conv_term(fieldT,temp,deriv=(0,1),
                              deriv_field = self.deriv_field,
                              dealias= self.dealias )

        conv += conv_term(fieldT,self.temp_bc,deriv=(0,1),
                              deriv_field = self.deriv_field,
                              dealias= self.dealias )

        return conv

    def update_NS(self):
        # -- Calculate Residual of Navier-Stokes
        self.NS.U.vhat[:] = self.U.vhat[:]
        self.NS.V.vhat[:] = self.V.vhat[:]
        self.NS.T.vhat[:] = self.T.vhat[:]
        self.NS.update()
        self.NS.U.vhat[:] = (self.NS.U.vhat[:]-self.U.vhat[:]) / self.dt
        self.NS.V.vhat[:] = (self.NS.V.vhat[:]-self.V.vhat[:]) / self.dt
        self.NS.T.vhat[:] = (self.NS.T.vhat[:]-self.T.vhat[:]) / self.dt

        # -- Smooth fields (nabla^2 TA = T)
        rhs  = self.nabla_U.solve_rhs(galerkin_to_cheby(self.NS.U.vhat,self.NS.U))
        self.UA.vhat[:] = self.nabla_U.solve_lhs(rhs)
        rhs  = self.nabla_V.solve_rhs(galerkin_to_cheby(self.NS.V.vhat,self.NS.V))
        self.VA.vhat[:] = self.nabla_V.solve_lhs(rhs)
        rhs  = self.nabla_T.solve_rhs(galerkin_to_cheby(self.NS.T.vhat,self.NS.T))
        self.TA.vhat[:] = self.nabla_T.solve_lhs(rhs)


        # self.VA.backward()
        # fig,ax = plt.subplots()
        # ax.contourf(self.xx,self.yy,self.VA.v)
        # ax.set_title("A")
        # ax.set_aspect(1)
        # plt.show()

    def update_U(self,stage):
        self.rhs[:] = 0.

        # Pressure term
        dpdx = grad(self.pres,deriv=(1,0),return_field=False)
        self.rhs -= self.a[stage]*dpdx

        # Non-Linear Convection
        self.rhs += self.b[stage]*self.NS.conv_term(self.UA,self.ux,self.uz)
        self.rhs += self.b[stage]*self.conv_term_adj_ux(self.UA,self.VA,self.TA,
            self.ux,self.uz,self.temp)
        if self.c[stage] != 0:
            self.rhs += self.c[stage]*self.NS.conv_term(self.UA,self.ux_old,self.uz_old)
            self.rhs += self.c[stage]*self.conv_term_adj_ux(self.UA,self.VA,self.TA,
                self.ux_old,self.uz_old,self.temp_old)

        # Diffusion
        self.rhs += self.a[stage]*self.nu*galerkin_to_cheby(self.NS.U.vhat,self.U)

        # Update
        self.U.vhat[:] += self.dt*cheby_to_galerkin(self.rhs,self.U)

    def update_V(self,stage):
        self.rhs[:] = 0.

        # Pressure term
        dpdz = grad(self.pres,deriv=(0,1),return_field=False)
        self.rhs -= self.a[stage]*dpdz

        # Non-Linear Convection
        self.rhs += self.b[stage]*self.NS.conv_term(self.VA,self.ux,self.uz)
        self.rhs += self.b[stage]*self.conv_term_adj_uz(self.UA,self.VA,self.TA,
            self.ux,self.uz,self.temp)
        if self.c[stage] != 0:
            self.rhs += self.c[stage]*self.NS.conv_term(self.VA,self.ux_old,self.uz_old)
            self.rhs += self.c[stage]*self.conv_term_adj_uz(self.UA,self.VA,self.TA,
                self.ux_old,self.uz_old,self.temp_old)

        # Diffusion
        self.rhs += self.a[stage]*self.nu*galerkin_to_cheby(self.NS.V.vhat,self.V)

        # Update
        self.V.vhat[:] += self.dt*cheby_to_galerkin(self.rhs,self.V)

    def update_T(self,stage):
        self.rhs[:] = 0.

        # Non-Linear Convection
        self.rhs += self.b[stage]*self.NS.conv_term(self.TA,self.ux,self.uz)
        if self.c[stage] != 0:
            self.rhs += self.c[stage]*self.NS.conv_term(self.TA,self.ux_old,self.uz_old)

        # Diffusion
        self.rhs += self.a[stage]*self.kappa*galerkin_to_cheby(self.NS.T.vhat,self.T)

        # Buoyancy
        self.rhs += self.a[stage]*galerkin_to_cheby(self.VA.vhat,self.VA)

        # Update
        self.T.vhat[:] += self.dt*cheby_to_galerkin(self.rhs,self.T)

    def update_P(self,div,singular=True):
        rhs = self.solver_P.solve_rhs(div)
        self.P.vhat[:] = self.solver_P.solve_lhs(rhs)
        if singular: self.P.vhat[0,0] = 0

    def update_pres(self,div,stage):
        #pres.vhat -=  1.0*self.nu*div*self.beta#*self.dt*self.a[stage]#self.dt*self.a[stage]*
        self.pres.vhat +=  galerkin_to_cheby(self.P.vhat,self.P)/(self.dt*self.a[stage])

    def update_velocity(self,p,u,v,fac=1.0):

        dpdx = grad(p,deriv=(1,0),return_field=False)
        dpdz = grad(p,deriv=(0,1),return_field=False)

        u.vhat -= cheby_to_galerkin(dpdx*fac,u)
        v.vhat -= cheby_to_galerkin(dpdz*fac,v)

    def callback(self):
        # -- Divergence
        print("Divergence   : {:4.2e}".format(
        np.linalg.norm(self.NS.divergence_velocity(self.U,self.V))))
        print("Divergence NS: {:4.2e}".format(
        np.linalg.norm(self.NS.divergence_velocity(self.NS.U,self.NS.V))))

        # -- Norm
        print(" |U| = {:5.2e}".format(np.linalg.norm(self.NS.U.vhat)))
        print(" |V| = {:5.2e}".format(np.linalg.norm(self.NS.V.vhat)))
        print(" |T| = {:5.2e}".format(np.linalg.norm(self.NS.T.vhat)))

    def update(self):
        self.ux_old,self.uz_old,self.temp_old = 0,0,0
        for rk in range(self.nstage):

            # Convection velocity
            if self.dealias:
                self.ux = self.U.dealias.backward(self.U.vhat)
                self.uz = self.V.dealias.backward(self.V.vhat)
                self.temp = self.T.dealias.backward(self.T.vhat)
            else:
                self.ux = self.U.backward(self.U.vhat)
                self.uz = self.V.backward(self.V.vhat)
                self.temp = self.T.backward(self.T.vhat)

            # -- Residual NS
            self.update_NS()

            # -- Ux
            self.update_U(stage=rk)

            # -- Uz
            self.update_V(stage=rk)

            # -- Pressure
            div = self.NS.divergence_velocity(self.U,self.V)
            self.update_P(div)
            self.update_pres(div,stage=rk)

            # -- Update Velocity
            self.update_velocity(self.P,self.U,self.V)

            # -- Temp
            self.update_T(stage=rk)

            # -- Save old velocities
            self.ux_old,self.uz_old,self.temp_old = self.ux,self.uz,self.temp

shape = (64,64)
Pr = 1
Ra = 5e3

NSA = NavierStokesAdjoint(shape=shape,dt=0.1,tsave=1.0,
nu=nu(Ra/2**3,Pr),kappa=kappa(Ra/2**3,Pr),
dealias=True,integrator="eu",beta=1.0)

NSA.NS.set_temperature(amplitude=0.2)
NSA.NS.iterate(100.)
NSA.U.vhat[:] = NSA.NS.U.vhat[:]
NSA.V.vhat[:] = NSA.NS.V.vhat[:]
NSA.T.vhat[:] = NSA.NS.T.vhat[:]
NSA.plot()

NSA.iterate(100.0)
#NSA.update()
#NSA.update()
NSA.plot()

# Animate
for i,v in enumerate(NSA.T.V):
        if NSA.T.V[i][0,0] < 0.1:
            NSA.T.V[i] += NSA.NS.Tbc.v

anim = NSA.T.animate(NSA.T.x,duration=4,wireframe=False)
plt.show()
