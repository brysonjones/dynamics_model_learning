
import torch
from torch.autograd.functional import jacobian, hessian


def solve_euler_lagrange(lagrange_fcn, state):
    '''
    :param lagrange_fcn: any generic lagrange calculating function for a
                         dynamical system. Input must only be state, and
                         output must only be a single scalar representing
                         the calculated energy of the system
    :param state: state vector of the system
    :return: state_dot: calculated derivative of the input state, which can
                        be used to integrate and simulate system dynamics
    '''
    n = state.shape[0] // 2  # Get number of generalized coords
    xv = torch.autograd.Variable(state, requires_grad=True)
    xv = torch.flatten(xv)
    y, yt = torch.split(xv, 2, dim=0)

    # The hessian/jacobian are calculated w.r.t all variables in xv
    # so select only relevant parts using slices
    A = hessian(lagrange_fcn, xv, create_graph=True)[n:, n:]
    B = torch.squeeze(jacobian(lagrange_fcn, xv, create_graph=True))[:n]
    C = hessian(lagrange_fcn, xv, create_graph=True)[n:, :n]

    ytt = torch.linalg.solve(A, (B - C @ yt).T)

    state_dot = torch.cat([yt, torch.squeeze(ytt)])

    return state_dot
