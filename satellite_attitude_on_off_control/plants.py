from numba import njit


@njit
def rotational_dynamics(f, FD, I):

    return (f - FD) / I