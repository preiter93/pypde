from navier import rbc2d, rbc2d_adj
from pypde import *
n = 20
base = Base(20, "CH")
x = np.linspace(-1,1,n)
x = base.x
print(x)
f = np.sin(x) + 2
f = 2*x**2 - 1
print(f)

a0 = base.forward_fft(f)
print(a0)
u0 = base.backward_fft(a0)
print(u0)
