from numba import njit


@njit
def actuator_dynamics(f, u, FM, TC):
    return (FM * u - f) / TC