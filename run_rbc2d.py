from pypde import *
from example import rbc2d
import matplotlib.pyplot as plt
initplot()

def avg_x(f,dx):
    return np.sum(f*dx[:,None],axis=0)/np.sum(dx)


shape = (64,64)

Pr = 1
Ra = np.logspace(5,6,10)
Ra = [5e3]
Nu = []
Lx = 1.
for R in Ra:
    r = R/2**3
    nu = np.sqrt(Pr/r)
    kappa = np.sqrt(1/Pr/r)

    # -- Solve Navier Stokes
    NS = rbc2d.NavierStokes(shape=shape,dt=0.1,tsave=1.0,nu=nu,kappa=kappa,
    dealias=True,integrator="rk3",beta=1.0,Lx=Lx)
    NS.set_temperature(m=1)
    NS.iterate(10.0)
    #NS.write()

    # -- Animate and plot 
    NS.plot()
    NS.animate()

    # -- Get Geometry
    x,y = NS.T.x,NS.T.y
    dx,dy = NS.T.dx, NS.T.dy
    xx,yy = np.meshgrid(x,y,indexing="ij")

    # -- Evaluate Nu
    Nuz,Nuv = NS.eval_Nu()
    Nu.append(Nuz)
