from numba import njit


@njit
def RK4(f, t, x, dt, params):

    k1 = f(t, x, params)
    k2 = f(t + dt/2, x + dt*k1/2, params)
    k3 = f(t + dt/2, x + dt*k2/2, params)
    k4 = f(t + dt, x + dt*k3, params)

    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)