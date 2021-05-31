import sys
sys.path.append("./")
from pypde import *
import matplotlib.pyplot as plt

def nu(Ra,Pr):
    return np.sqrt(Pr/Ra)

def kappa(Ra,Pr):
    return  np.sqrt(1/Pr/Ra)

def Ra(nu,kappa,L=2):
    return 1/(nu*kappa)*L**3.

def Pr(nu,kappa):
    return nu/kappa

class NavierStokesBase():
    '''
    Some Base Functions for Navier--Stokes Simulations
    '''
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
        "Lx": 1.0,
    }

    def __init__(self,**kwargs):
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        self.Lz = 1.0

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


    def animate(self):
        #  Add inhomogeneous part
        T2 = Field(self.T.xs)
        T2.T = self.T.T
        for i,V in enumerate(self.T.V):
            T2.V.append(V+self.Tbc.v)

        anim = T2.animate(duration=4,x=self.x,y=self.y)
        anim.save('out/anim.gif', writer='imagemagick', fps=20)
        plt.show()

    def eval_Nu(self):
        from pypde.field_operations import eval_Nu,eval_Nuvol
        Nuz = eval_Nu(self.T,self.deriv_field)
        Nuv = eval_Nuvol(self.T,self.V,self.kappa,self.deriv_field,Tbc=self.Tbc)
        return Nuz,Nuv

    def interpolate(self,NS_old,spectral=True):
        self.field.interpolate(NS_old.field)

    def write(self,leading_str="",add_time=True):
        dict = {"nu": self.nu,"kappa": self.kappa,
        "Ra":Ra(self.nu,self.kappa),"Pr": Pr(self.nu,self.kappa)}
        self.field.write(leading_str=leading_str,add_time=add_time,dict=dict)

    def read(self,leading_str="",add_time=True):
        dict = {"nu": self.nu,"kappa": self.kappa}
        self.field.read(leading_str=leading_str,add_time=add_time,dict=dict)
        self.time = self.field[0].t # Update time

    def save(self):
        self.field.save()


class NavierStokesSteadyState():
    '''
    Add on for Navier-Stokes class. 
    Calculate steaday state solutions using the LGMRES algorithm.
    '''

    def solve_steady_state(self,X0=None,maxiter=300,disp=True,tol=1e-4):
        '''
        Solve steady state using scipy's LGMRES algorithm
        '''
        from scipy import optimize
        ''' Solve steady state '''
        options={"maxiter": maxiter, "disp": disp, "fatol": tol}
        if X0 is None:
            X0 = self.vectorify()
        sol = optimize.root(steady_fun, X0, args=(self,), 
            method="krylov",options=options)
        return sol

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