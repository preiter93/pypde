import numpy as np
from .field import Field, FieldBC
from .bases.spectralbase import Base

# -------------------------------------------------------
#                Mathematical operations
# -------------------------------------------------------
def grad(field, deriv, return_field=False, scale=None):
    """
    Find derivative of field

    Example:

    # Set field
    N,M = 40,30
    xbase = Base(N,"CD")
    ybase = Base(M,"CN")
    field = Field( [xbase,ybase] )
    xx,yy = np.meshgrid(field.x,field.y,indexing="ij")

    f = np.sin(np.pi* xx)*np.sin(np.pi*yy)
    field.v = f
    field.forward()

    # Get derivative
    deriv_field = grad(field,deriv=(1,0),return_field=True)
    deriv_field.backward()

    from pypde.plot.wireframe import plot
    plot(xx,yy,field.v)
    plot(xx,yy,deriv_field.v)
    """
    assert isinstance(field, (Field, FieldBC))
    if isinstance(deriv, int):
        deriv = (deriv,)  # to tuple
    assert field.ndim == len(deriv)

    dvhat = field.vhat

    for axis in range(field.ndim):
        dvhat = field.derivative(dvhat, deriv[axis], axis=axis)
        # Rescale
        if scale is not None:
            assert len(scale) == field.ndim
            dvhat /= scale[axis] ** deriv[axis]

    if return_field:
        field_deriv = Field([field.xs[i].family for i in range(field.ndim)])
        field_deriv.vhat = dvhat
        return field_deriv
    else:
        return dvhat


def cheby_to_galerkin(uhat, galerkin_field):
    for axis in range(uhat.ndim):
        if axis == 0:
            uhat = galerkin_field.xs[axis].from_chebyshev(uhat)
        else:
            uhat = np.swapaxes(uhat, 0, axis)
            uhat = galerkin_field.xs[axis].from_chebyshev(uhat)
            uhat = np.swapaxes(uhat, 0, axis)
    return uhat


def galerkin_to_cheby(vhat, galerkin_field):
    for axis in range(vhat.ndim):
        if axis == 0:
            vhat = galerkin_field.xs[axis].to_chebyshev(vhat)
        else:
            vhat = np.swapaxes(vhat, 0, axis)
            vhat = galerkin_field.xs[axis].to_chebyshev(vhat)
            vhat = np.swapaxes(vhat, 0, axis)
    return vhat


def conv_term(v_field, u, deriv, deriv_field=None, dealias=False, scale=None):
    """
    Calculate
        u*dvdx

    Input
        v_field: class Field
            Contains field variable vhat in spectral space
        u:  ndarray
            (Dealiased) velocity field in physical space
        deriv: tuple
            (1,0) for partial_x, (0,1) for partial_z
        deriv_field: field (optional)
            Field (space) where derivatives life
        dealias: bool (optional)
            Dealias convective term. In this case, input ux and
            uz must already be dealiased and deriv_field must
            be initialized with ,dealias=3/2
        scale: tuple float
            Scale physical domain size

    Return
        Field of (dealiased) convective term in physical space
        Transform to spectral space via conv_field.forward()
    """
    assert isinstance(v_field, Field), "v_field must be instance Field"

    if deriv_field is None:
        if dealias:
            deriv_field = Field(
                [
                    Base(v_field.shape[0], "CH", dealias=3 / 2),
                    Base(v_field.shape[1], "CH", dealias=3 / 2),
                ]
            )
        else:
            deriv_field = Field(
                [Base(v_field.shape[0], "CH"), Base(v_field.shape[1], "CH")]
            )

    # dvdx
    vhat = grad(v_field, deriv, return_field=False, scale=scale)
    if dealias:
        dvdx = deriv_field.dealias.backward(vhat)
    else:
        dvdx = deriv_field.backward(vhat)
    return dvdx * u


def convective_term(
    v_field, ux, uz, deriv_field=None, add_bc=None, dealias=False, scale=None
):
    """
    Calculate
        ux*dvdx + uz*dvdz

    Input
        v_field: class Field
            Contains field variable vhat in spectral space
        ux,uz:  ndarray
            (Dealiased) velocity fields in physical space
        deriv_field: field (optional)
            Field (space) where derivatives life
        add_bc: ndarray (optional)
            Additional term (physical space), which is added
            before forward transform.
        dealias: bool (optional)
            Dealias convective term. In this case, input ux and
            uz must already be dealiased and deriv_field must
            be initialized with ,dealias=3/2
        scale: tuple float
            Scale physical domain size

    Return
        Field of (dealiased) convective term in spectral space
        Transform to spectral space via conv_field.forward()
    """
    conv = conv_term(v_field, ux, (1, 0), deriv_field, dealias, scale=scale)
    conv += conv_term(v_field, uz, (0, 1), deriv_field, dealias, scale=scale)

    if add_bc is not None:
        conv += add_bc

    if dealias:
        return deriv_field.dealias.forward(conv)

    return deriv_field.forward(conv)


# -------------------------------------------------------
#                General operations
# -------------------------------------------------------
def avg_x(f, dx):
    return np.sum(f * dx[:, None], axis=0) / np.sum(dx)


def avg_vol(f, dx, dy):
    favgx = np.sum(f * dx[:, None], axis=0) / np.sum(dx)
    return np.sum(favgx * dy) / np.sum(dy)


def interpolate(Field_old, Field_new, spectral=True):
    """
    Interpolate from field F_old to Field F_new
    performed in spectral space
    Must be of same dimension

    Input
        Field_old: Field
        Field_new: Field
        spectral: bool (optional)
            if True, perform interpolation in spectral space
            if False, perform it in physical space
    """
    if spectral:
        F_old = Field_old.vhat
        F_new = Field_new.vhat
    else:
        F_old = Field_old.v
        F_new = Field_new.v

    if F_old.ndim != F_new.ndim:
        raise ValueError("Field must be of same dimension!")

    shape_max = [max(i, j) for i, j in zip(F_old.shape, F_new.shape)]
    sl_old = tuple([slice(0, N, None) for N in F_old.shape])
    sl_new = tuple([slice(0, N, None) for N in F_new.shape])
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
