
import torch

def rk4_step(fcn, xk, step_size):

    f1 = fcn(xk)
    f2 = fcn(xk + 0.5 * step_size * f1)
    f3 = fcn(xk + 0.5 * step_size * f2)
    f4 = fcn(xk + step_size * f3)

    xn = xk + (step_size / 6) * (f1 + 2 * f2 + 2 * f3 + f4)

    return xn

# TODO: add more integrators
