import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(xx,yy,f):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(xx, yy, f,cmap='viridis', edgecolor='k')
    ax.set_title('Surface plot')
    plt.show()