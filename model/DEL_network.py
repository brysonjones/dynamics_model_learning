import numpy as np
from scipy.optimize import root

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
        self.num_outputs = int(((num_states-1)**2 + (num_states-1)) / 2)

        # input layer
        self.model_layers.append(torch.nn.Linear(num_states, hidden_list[0]))
        # self.model_layers.append(torch.nn.Dropout())
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            # self.model_layers.append(torch.nn.Dropout())
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        # self.model_layers.append(torch.nn.Dropout())
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], self.num_outputs))

    def forward(self, q):
        """
        XXXX
        """
        diag_bias = 1e-3  # TODO: determine if this should be learned
        cholesky_raw = q
        for layer in self.model_layers:
            cholesky_raw = layer(cholesky_raw)

        diag = cholesky_raw[0:self.num_states-1] + diag_bias
        lower_tri = cholesky_raw[self.num_states-1:]

        cholesky = torch.diag(diag)
        lower_tri_indices = np.tril_indices(self.num_states-1, -1)
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
        # self.model_layers.append(torch.nn.Dropout())
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            # self.model_layers.append(torch.nn.Dropout())
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        # output is always one from this network because it is calculating system energy
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 1))
        # self.model_layers.append(torch.nn.Dropout())
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
        # self.model_layers.append(torch.nn.Dropout())
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            # self.model_layers.append(torch.nn.Dropout())
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        # output is always one from this network because it is calculating system energy
        # self.model_layers.append(torch.nn.Dropout())
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 6))

    def forward(self, q, q_next, h):
        r1 = q[0:3]
        Q1 = q[3:7]
        r2 = q_next[0:3]
        Q2 = q_next[3:7]
        omega = 2 / h * (H.T @ L(Q1).T @ Q2)
        Q_mid = Q1 + 0.5 * h * (0.5 * L(Q1) @ H @ omega)  # TODO: Determine if this is the best implementation

        r_mid = (r1 + r2) / 2
        r_mid_vel = (r2 - r1) / h

        out = torch.cat((r_mid, Q_mid, r_mid_vel, omega))
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
        # self.model_layers.append(torch.nn.Dropout())
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            # self.model_layers.append(torch.nn.Dropout())
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        # self.model_layers.append(torch.nn.Dropout())
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 6*num_control_inputs))

    def forward(self, q, q_next, h):
        """
        XXXX
        """
        r1 = q[0:3]
        Q1 = q[3:7]
        r2 = q_next[0:3]
        Q2 = q_next[3:7]
        omega = 2 / h * (H.T @ L(Q1).T @ Q2)
        Q_mid = Q1 + 0.5 * h * (0.5 * L(Q1) @ H @ omega) # TODO: Determine if this is the best implementation

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
        num_control_inputs = 4
        self.h = 1/400  # Time Step of Dataset
        hidden_list = [64, 64, 64, 64, 64]
        self.M_ = None

        self.massMatrix = MassMatrixNetwork(num_pose_states, hidden_list)
        self.potentialEnergy = PotentialEnergyNetwork(num_pose_states, hidden_list)
        self.dissipativeForces = DissipativeForceNetwork(num_states, hidden_list)
        self.controlJacobian = ControlInputJacobianNetwork(num_pose_states, num_control_inputs, hidden_list)

    def discrete_lagrangian(self, q, q_next):
        r1 = q[0:3]
        Q1 = q[3:7]
        r2 = q_next[0:3]
        Q2 = q_next[3:7]
        omega = 2 / self.h * (H.T @ L(Q1).T @ Q2)
        Q_mid = Q1 + 0.5 * self.h * (0.5 * L(Q1) @ H @ omega)  # TODO: Determine if this is the best implementation

        r_mid = (r1 + r2) / 2
        r_mid_vel = (r2 - r1) / self.h

        # calculate system parameters and values
        self.M_ = self.massMatrix(torch.cat((r_mid, Q_mid)))
        m = 0.752 * torch.eye(3)
        out_trans = self.h * (1/2 * r_mid_vel.T @ m @ r_mid_vel - self.potentialEnergy(torch.cat((r_mid, Q_mid))))
        out_rot = self.h / 2 * ((2/self.h * H.T @ L(Q1) @ Q2).T @ self.M_[3:, 3:] @ (2/self.h * H.T @ L(Q1) @ Q2))

        return out_trans + out_rot

    def DEL(self, q1, q2, q3, u1, u2, u3):
        q2 = torch.autograd.Variable(q2, requires_grad=True)

        diss_forces1 = self.dissipativeForces(q1, q2, self.h)
        diss_forces2 = self.dissipativeForces(q2, q3, self.h)
        control_forces1 = self.controlJacobian(q1, q2, self.h) @ ((u1 + u2) / 2)
        control_forces2 = self.controlJacobian(q2, q3, self.h) @ ((u2 + u3) / 2)

        L1_fcn = lambda q_: self.discrete_lagrangian(q1, q_)
        L2_fcn = lambda q_: self.discrete_lagrangian(q_, q3)

        DL1 = jacobian(L1_fcn, q2, create_graph=True)
        DL2 = jacobian(L2_fcn, q2, create_graph=True)

        DEL = DL1@G_(q2) + DL2@G_(q2) + self.h/2 * (diss_forces1 + diss_forces2 + control_forces1 + control_forces2)

        return DEL

    def step(self, q1, q2, u1, u2, u3):
        self.massMatrix.eval()
        self.potentialEnergy.eval()
        self.dissipativeForces.eval()
        self.controlJacobian.eval()

        q1 = torch.tensor(q1).float()
        q2 = torch.tensor(q2).float()
        u1 = torch.tensor(u1).float()
        u2 = torch.tensor(u2).float()
        u3 = torch.tensor(u3).float()

        q3_guess = q2 + (q2 - q1)
        q3_guess = torch.autograd.Variable(q3_guess, requires_grad=True)
        fcn = lambda q_: self.DEL(q1, q2, q_, u1, u2, u3)

        # use Newton's Method to find true zero
        while True:
            e = torch.squeeze(self.DEL(q1, q2, q3_guess, u1, u2, u3))
            if torch.linalg.norm(e) < 1:
                break
            res_jac = torch.squeeze(jacobian(fcn, q3_guess, create_graph=True, strict=True))
            with torch.no_grad():
                res_jac_trans = res_jac[0:3, 0:3]
                res_jac_rot = res_jac[3:6, 3:7] @ G(q3_guess[3:7])
                phi = torch.linalg.solve(-res_jac_rot.T, e[3:6])
                q3_guess[0:3] = q3_guess[0:3] - torch.linalg.solve(res_jac_trans, e[0:3])
                q3_guess[3:7] = L(q3_guess[3:7]) @ rho(phi)

        print(q3_guess)
        q3 = q3_guess

        return q3.detach().numpy()

    def forward(self, x):
        """
        XXXX
        """
        q1 = x[0, [0, 1, 2, 9, 6, 7, 8]]
        q2 = x[1, [0, 1, 2, 9, 6, 7, 8]]
        q3 = x[2, [0, 1, 2, 9, 6, 7, 8]]
        u1 = x[0, 13:17]
        u2 = x[1, 13:17]
        u3 = x[2, 13:17]

        mu = 0.01
        alpha = 1e-1
        return torch.linalg.norm(self.DEL(q1, q2, q3, u1, u2, u3)) - mu*torch.det(self.M_ - alpha*torch.eye(6))
