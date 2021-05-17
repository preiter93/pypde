from pypde import *
import matplotlib.pyplot as plt
from example import rbc2d
initplot()

def avg_x(f,dx):
    return np.sum(f*dx[:,None],axis=0)/np.sum(dx)


shape = (64,64)

Pr = 1
Ra = np.logspace(5,6,10)
Ra = [5e3]
Nu = []
for R in Ra:
    r = R/2**3
    nu = np.sqrt(Pr/r)
    kappa = np.sqrt(1/Pr/r)

    # -- Solve Navier Stokes
    NS = rbc2d.NavierStokes(shape=shape,dt=0.009,tsave=0.01,nu=nu,kappa=kappa,
    dealias=True,integrator="rk3",beta=1.0)

    NS.iterate(1.0)

    # -- Get Geometry
    x,y = NS.T.x,NS.T.y
    dx,dy = NS.T.dx, NS.T.dy
    xx,yy = np.meshgrid(x,y,indexing="ij")

    # -- Evaluate Nu
    T = NS.T.V[-1]

    Field = NS.deriv_field
    That = Field.forward(T)
    dThat = Field.derivative(That, 1, axis=1)
    dT = Field.backward(dThat)

    dTavg = avg_x(dT,Field.dx)
    Nu_bot = - dTavg[0]/0.5
    Nu_top = - dTavg[-1]/0.5
    print("Nubot: {:6.2f}".format(Nu_bot))
    print("Nutop: {:6.2f}".format(Nu_top))

    Nu.append(Nu_bot)
