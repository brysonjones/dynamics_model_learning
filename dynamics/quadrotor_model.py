
import numpy as np
import torch

class QuadRotor:
    """A class to represent a quad rotor and all of its system properties"""

    def __init__(self, m_matrix, J_matrix, l_vec):
        self.m = torch.tensor(m_matrix)
        self.J = torch.tensor(J_matrix)
        self.lengths = torch.tensor(l_vec)
        self.g = torch.tensor(9.81)

    def _newton_euler_dynamics(self, state):
        """
        :param state: []
        :return state_dot:
        """
        state_dot = None

        return state_dot

    def _network_dynamics(self, state):
        """
        :param state: []
        :return state_dot:
        """
        state_dot = None

        return state_dot

if __name__ == "__main__":

    pass



