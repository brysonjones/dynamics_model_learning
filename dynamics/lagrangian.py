
import torch
from torch.autograd.functional import jacobian, hessian


def solve_euler_lagrange(lagrange_fcn, external_force_fcn, pose, pose_t, ext_input):
    '''
    :param lagrange_fcn: any generic lagrange calculating function for a
                         dynamical system. Input must only be state, and
                         output must only be a single scalar representing
                         the calculated energy of the system
    :param external_force_fcn: function for calculating external forces of
                               the system
    :param pose: pose vector of the system
    :param pose_t: derivative of pose vector of the system
    :param ext_input: input variables contributing to external forces
    :return: state_dot: calculated derivative of the input state, which can
                        be used to integrate and simulate system dynamics
    '''
    pose = torch.autograd.Variable(pose, requires_grad=True)
    pose_t = torch.autograd.Variable(pose_t, requires_grad=True)
    ext_input = torch.autograd.Variable(ext_input, requires_grad=True)
    # xv = torch.flatten(xv)
    # print(xv)
    # y, yt = torch.split(xv, 2, dim=0)

    # The hessian/jacobian are calculated w.r.t all variables in xv
    # so select only relevant parts using slices
    A = hessian(lagrange_fcn, xv, create_graph=True)[n:, n:]
    B = torch.squeeze(jacobian(lagrange_fcn, xv, create_graph=True))[:n]
    C = hessian(lagrange_fcn, xv, create_graph=True)[n:, :n]
    F = external_force_fcn(ext_input)

    ytt = torch.linalg.solve(A, (B - C @ yt).T)

    state_dot = torch.cat([yt, torch.squeeze(ytt)])

    return state_dot
