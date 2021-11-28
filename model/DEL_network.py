import numpy as np
import torch
from torch.autograd.functional import jacobian, hessian
from torch.autograd import grad

import sys
sys.path.append("../dynamics")

from dynamics.lagrangian import *
from dynamics.quaternion import *

class MassMatrixNetwork(torch.nn.Module):
    def __init__(self, num_states, hidden_list):
        """
        Neural Network used to approximate a lagrangian systems mass matrix
        - It should be noted that this mass matrix is enforced to be positive definite,
          to ensure that all eigenvalues are positive and the matrix is invertible
        """
        super(MassMatrixNetwork, self).__init__()
        self.hyperparams = None
        self.model_layers = torch.nn.ModuleList()
        self.num_states = num_states
        self.num_outputs = (num_states**2 + num_states) / 2

        # input layer
        self.model_layers.append(torch.nn.Linear(num_states, hidden_list[0]))
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], num_states))

    def forward(self, q):
        """
        XXXX
        """
        diag_bias = 1.0  # TODO: determine if this should be learned
        cholesky_raw = q
        for layer in self.diag_model_layers:
            cholesky_raw = layer(cholesky_raw)

        diag = cholesky_raw[0:self.num_states] + diag_bias
        lower_tri = cholesky_raw[self.num_states:]

        cholesky = torch.diag(diag)
        lower_tri_indices = np.tril_indices(len(q), -1)
        cholesky[lower_tri_indices] = lower_tri
        M = cholesky @ cholesky.T

        return M


class PotentialEnergyNetwork(torch.nn.Module):
    def __init__(self, q_size, hidden_list):
        """
        Neural Network used to approximate a lagrangian systems potential energy
        """
        super(PotentialEnergyNetwork, self).__init__()
        self.hyperparams = None
        self.model_layers = torch.nn.ModuleList()

        # input layer - Lagrange
        self.model_layers.append(torch.nn.Linear(q_size, hidden_list[0]))
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        # output is always one from this network because it is calculating system energy
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 1))
        self.model_layers.append(torch.nn.Softplus())

    def forward(self, q):
        for layer in self.model_layers:
            q = layer(q)

        return q


class DissipativeForceNetwork(torch.nn.Module):
    def __init__(self, D_in, hidden_list):
        """
        Neural Network used to the represent dissipative forces such as drag, friction, etc
        """
        super(DissipativeForceNetwork, self).__init__()
        self.hyperparams = None
        self.model_layers = torch.nn.ModuleList()

        # input layer - Lagrange
        self.model_layers.append(torch.nn.Linear(D_in, hidden_list[0]))
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        # output is always one from this network because it is calculating system energy
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 6))

    def forward(self, q, q_dot):
        out = torch.cat((q, q_dot))
        for layer in self.model_layers:
            out = layer(out)

        return out



class ControlInputJacobianNetwork(torch.nn.Module):
    def __init__(self, D_in, num_control_inputs, hidden_list):
        """
        Neural Network used to represent the control input Jacobian for the system
        """
        super(ControlInputJacobianNetwork, self).__init__()
        self.hyperparams = None
        self.num_control_inputs = num_control_inputs
        self.model_layers = torch.nn.ModuleList()

        # input layer
        self.model_layers.append(torch.nn.Linear(D_in, hidden_list[0]))
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 6*num_control_inputs))

    def forward(self, q, q_next, h):
        """
        XXXX
        """
        q = torch.autograd.Variable(q, requires_grad=True)
        q_next = torch.autograd.Variable(q_next, requires_grad=True)
        r1 = q[1:3]
        Q1 = q[4:7]
        r2 = q_next[1:3]
        Q2 = q_next[4:7]
        omega = 2 / h * (H.T @ L(Q1).T @ Q2)
        Q_mid = Q1 + 0.5 * h * (0.5 * L(Q1) * H * omega) # TODO: Determine if this is the best implementation

        r_mid = (r1 + r2) / 2
        q_mid = torch.cat((r_mid, Q_mid))

        out = q_mid

        for layer in self.model_layers:
            out = layer(out)

        return torch.reshape(out, (6, self.num_control_inputs))

class DELNetwork(torch.nn.Module):
    def __init__(self, num_states):
        """
        Neural Network used to approximate the discrete Euler-Lagrange of the system
        """
        super(DELNetwork, self).__init__()
        num_pose_states = 7
        num_velocity_states = 6
        num_control_inputs = 4

        self.massMatrix = MassMatrixNetwork(num_pose_states, [32, 32, 32])
        self.potentialEnergy = PotentialEnergyNetwork(num_pose_states, [32, 32, 32])
        self.dissipativeForces = DissipativeForceNetwork(num_states, [32, 32, 32])
        self.controlJacobian = ControlInputJacobianNetwork(num_pose_states, num_control_inputs, [32, 32, 32])

    def discrete_lagrangian(self, q, q_next, h):
        q = torch.autograd.Variable(q, requires_grad=True)
        q_next = torch.autograd.Variable(q_next, requires_grad=True)
        r1 = q[1:3]
        Q1 = q[4:7]
        r2 = q_next[1:3]
        Q2 = q_next[4:7]
        omega = 2 / h * (H.T @ L(Q1).T @ Q2)
        Q_mid = Q1 + 0.5 * h * (0.5 * L(Q1) * H * omega) # TODO: Determine if this is the best implementation

        r_mid = (r1 + r2) / 2
        r_mid_vel = (r2 - r1) / h

        # calculate system parameters and values
        M_ = self.massMatrix(torch.cat((r_mid, Q_mid)))

        out_trans = 1 / 2 * r_mid_vel.T @ M_[:3, :3] @ r_mid_vel - self.potentialEnergy(torch.cat((r_mid, Q_mid)))
        out_rot = h / 2 * ((2/h * H.T @ L(Q1) @ Q2).T @ M_[3:, 3:] @ (2/h * H.T @ L(Q1) @ Q2))

        return torch.cat((out_trans, out_rot))

    def forward(self, q1, q2, q3, u1, u2, u3, h):
        """
        XXXX
        """
        u1 = torch.autograd.Variable(u1, requires_grad=True)
        u2 = torch.autograd.Variable(u2, requires_grad=True)
        u3 = torch.autograd.Variable(u3, requires_grad=True)

        dl1 = self.discrete_lagrangian(q1, q2, h)
        dl2 = self.discrete_lagrangian(q2, q3, h)

        diss_forces1 = self.dissipativeForces(q1, q2, h)
        diss_forces2 = self.dissipativeForces(q2, q3, h)
        control_forces1 = self.controlJacobian(q1, q2, h) @ ((u1 + u2) / 2)
        control_forces2 = self.controlJacobian(q2, q3, h) @ ((u2 + u3) / 2)

        DL1 = grad(dl1, q2, retain_graph=True)[0]
        DL2 = grad(dl2, q2, retain_graph=True)[0]

        DEL = DL1 + DL2 + 0.5 * (diss_forces1 + diss_forces2 + control_forces1 + control_forces2)

        return DEL
