import numpy as np
from scipy.optimize import root

import torch
from torch.autograd.functional import jacobian, hessian
from torch.autograd import grad

import sys
sys.path.append("../dynamics")

from dynamics.lagrangian import *
from dynamics.quaternion import *

def calc_midpoint(q1, q2, h):
    r1 = q1[0:3]
    Q1 = q1[3:7]
    r2 = q2[0:3]
    Q2 = q2[3:7]
    omega = 2 / h * (H.T @ L(Q1).T @ Q2)
    Q_mid = Q1 + 0.5 * h * (0.5 * L(Q1) @ H @ omega)  # TODO: Determine if this is the best implementation

    r_mid = (r1 + r2) / 2
    r_mid_vel = (r2 - r1) / h

    return r_mid, r_mid_vel, Q_mid, omega

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
        self.model_layers.append(torch.nn.Dropout(0.15))
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            self.model_layers.append(torch.nn.Dropout(0.15))
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        self.model_layers.append(torch.nn.Dropout(0.15))
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], self.num_outputs))

    def forward(self, q):
        """
        XXXX
        """
        diag_bias = 1e-1  # TODO: determine if this should be learned
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
        self.model_layers.append(torch.nn.Dropout(0.3))
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            self.model_layers.append(torch.nn.Dropout(0.3))
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        # output is always one from this network because it is calculating system energy
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 1))
        self.model_layers.append(torch.nn.Dropout(0.3))
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
        self.model_layers.append(torch.nn.Dropout(0.3))
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            self.model_layers.append(torch.nn.Dropout(0.3))
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        # output is always one from this network because it is calculating system energy
        self.model_layers.append(torch.nn.Dropout(0.3))
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 6))

    def forward(self, r_mid, r_mid_vel, Q_mid, omega):

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
        self.model_layers.append(torch.nn.Dropout(0.3))
        self.model_layers.append(torch.nn.Tanh())
        # add all hidden layers
        for i in range(0, len(hidden_list) - 1):
            self.model_layers.append(torch.nn.Linear(hidden_list[i], hidden_list[i + 1]))
            self.model_layers.append(torch.nn.Dropout(0.3))
            self.model_layers.append(torch.nn.Tanh())

        # output layer
        self.model_layers.append(torch.nn.Dropout(0.3))
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], 6*num_control_inputs))

    def forward(self, r_mid, Q_mid):
        """
        XXXX
        """
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
        num_trans_states = 3
        num_pose_states = 7
        num_control_inputs = 4
        self.h = 1/400  # Time Step of Dataset
        hidden_list = [64, 64, 64, 64, 64]
        self.M_ = None

        self.massMatrix = MassMatrixNetwork(num_pose_states, hidden_list)
        self.potentialEnergy = PotentialEnergyNetwork(num_trans_states, hidden_list)
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
        J = torch.as_tensor([[0.00466178, 0, 0],
                             [0, 0.00466178, 0],
                             [0, 0, 0.0131855]])
        # m = self.M_[:3, :3]
        # J = self.M_[3:, 3:]

        out_trans = self.h * (1/2 * r_mid_vel.T @ m @ r_mid_vel - self.potentialEnergy(r_mid))
        out_rot = self.h / 2 * ((2/self.h * H.T @ L(Q1).T @ Q2).T @ J @ (2/self.h * H.T @ L(Q1).T @ Q2))

        return out_trans + out_rot

    def DEL(self, q1, q2, q3, u1, u2, u3):
        q2 = torch.autograd.Variable(q2, requires_grad=True)

        r_mid1, r_mid_vel1, Q_mid1, omega1 = calc_midpoint(q1, q2, self.h)
        r_mid2, r_mid_vel2, Q_mid2, omega2 = calc_midpoint(q2, q3, self.h)

        diss_forces1 = self.dissipativeForces(r_mid1, r_mid_vel1, Q_mid1, omega1)
        diss_forces2 = self.dissipativeForces(r_mid2, r_mid_vel2, Q_mid2, omega2)
        control_forces1 = self.controlJacobian(r_mid1, Q_mid1) @ ((u1 + u2) / 2)
        control_forces2 = self.controlJacobian(r_mid2, Q_mid2) @ ((u2 + u3) / 2)

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

        q3_guess = q2 + (q2 - q1)
        q3_guess = torch.autograd.Variable(q3_guess, requires_grad=True)
        fcn = lambda q_: self.DEL(q1, q2, q_, u1, u2, u3)

        # use Newton's Method to find true zero
        for i in range(15):
            e = torch.squeeze(self.DEL(q1, q2, q3_guess, u1, u2, u3))
            if torch.linalg.norm(e) < 1e-4:
                break
            res_jac = torch.squeeze(jacobian(fcn, q3_guess, create_graph=True, strict=True))
            with torch.no_grad():
                res_jac_trans = res_jac[0:3, 0:3]
                res_jac_rot = res_jac[3:6, 3:7] @ G(q3_guess[3:7])
                phi = torch.linalg.solve(-res_jac_rot.T, e[3:6])
                q3_guess[0:3] = q3_guess[0:3] - torch.linalg.solve(res_jac_trans, e[0:3])
                q3_guess[3:7] = L(q3_guess[3:7]) @ rho(phi)

        q3 = q3_guess

        return q3

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
        # return torch.linalg.norm(self.DEL(q1, q2, q3, u1, u2, u3)) - mu * torch.linalg.det(self.massMatrix(q2) - alpha*torch.eye(6))
        return torch.linalg.norm(self.DEL(q1, q2, q3, u1, u2, u3))