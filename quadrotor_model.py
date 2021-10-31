
import numpy as np

import torch
from torch.autograd.functional import jacobian, hessian

class QuadRotor:
    """A class to represent a quad rotor and all of its system properties"""

    def __init__(self, m_matrix, J_matrix, l_vec):
        self.m = torch.tensor(m_matrix)
        self.J = torch.tensor(J_matrix)
        self.lengths = torch.tensor(l_vec)
        self.g = torch.tensor(9.81)

    def _dynamics(self, state):
        """
        :param state: []
        :return state_dot:
        """
        state_dot = None

        return state_dot

    def _lagrangian(self, state):
        """
        :param state:
        :return lagrangian:
        """

        lagrangian = None

        return lagrangian

    def _autodiff(self, state):
        state = torch.tensor(state)

        n = state.shape[0] // 2  # Get number of generalized coords
        xv = torch.autograd.Variable(state, requires_grad=True)
        y, yt = torch.split(xv, 2, dim=0)

        # The hessian/jacobian are calculated w.r.t all variables in xv
        # so select only relevant parts using slices
        A = torch.inverse(hessian(self._lagrangian, xv, create_graph=True)[n:, n:])
        B = jacobian(self._lagrangian, xv, create_graph=True)[:, :n]
        C = hessian(self._lagrangian, xv, create_graph=True)[n:, :n]

        ytt = A @ (B - C @ yt).T

        xt = torch.cat([yt, torch.squeeze(ytt)])

        return xt

if __name__ == "__main__":

    m = np.array([[1],
                  [1]])
    J = np.array([[1],
                  [1]])
    l = np.array([[1],
                  [1]])

    quadRotor = QuadRotor(m, J, l)

    x0 = np.array([0.5, 0.9, -0.4, 0.1])

    xt = quadRotor._autodiff(state=x0)
    print("Autodiff:", xt)
    x_dot = quadRotor._dynamics(state=x0)
    print("Analytical:", xt)



