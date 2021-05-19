import numpy as np 

def avg_x(f,dx):
    return np.sum(f*dx[:,None],axis=0)/np.sum(dx)


def avg_vol(f,dx,dy):
    favgx =  np.sum(f*dx[:,None],axis=0)/np.sum(dx)
    return np.sum(favgx*dy)/np.sum(dy)

def eval_Nu(T,field):
     # -- Evaluate Nu
    T = T.v.copy()

    That = field.forward(T)
    dThat = field.derivative(That, 1, axis=1)
    dT = field.backward(dThat)

    dTavg = avg_x(dT,field.dx)
    Nu_bot = - dTavg[0]/0.5 + 1
    Nu_top = - dTavg[-1]/0.5 + 1
    print("Nubot: {:10.6e}".format(Nu_bot))
    print("Nutop: {:10.6e}".format(Nu_top))
    return (Nu_bot+Nu_top)/2. 


def eval_Nuvol(T,V,kappa,field,Tbc=None):
    T = T.v.copy() 
    if Tbc is not None:
        T += Tbc.v.copy()
    V = V.v.copy()
    
    That = field.forward(T)
    dThat = field.derivative(That, 1, axis=1)
    dT = field.backward(dThat)
    
    Nuvol = (T*V/kappa - dT)*2.0
    
    Nuvol = avg_vol(Nuvol,field.dx,field.dy)
    print("Nuvol: {:10.6e}".format(Nuvol))
    return Nuvol

def interpolate(Field_old,Field_new,spectral=True):
    '''
    Interpolate from field F_old to Field F_new
    performed in spectral space
    Must be of same dimension
    
    Input
        Field_old: Field
        Field_new: Field
        spectral: bool (optional)
            if True, perform interpolation in spectral space
            if False, perform it in physical space
    '''
    if spectral:
        F_old = Field_old.vhat
        F_new = Field_new.vhat
    else:
        F_old = Field_old.v
        F_new = Field_new.v
        
    if F_old.ndim != F_new.ndim:
        raise ValueError("Field must be of same dimension!")
        
    shape_max = [max(i,j) for i,j in zip(F_old.shape,F_new.shape)]
    sl_old = tuple([slice(0,N,None) for N in F_old.shape])
    sl_new = tuple([slice(0,N,None) for N in F_new.shape])
    # Create buffer which has max size, then return slice
    buffer = np.zeros(shape_max)
    buffer[sl_old] = F_old
    F_new[:] = buffer[sl_new]
    try:
        if spectral:
            Field_new.backward()
        else:
            Field_new.forward()
    except:
        print("Backward Transformation failed after interpolation.")