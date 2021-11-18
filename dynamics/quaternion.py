
import torch

H = torch.tensor([[torch.zeros(1,3)], [torch.eye(3)]])

#Some standard functions for dealing with rotation matrices and quaternions from the class notes
def hat(omega):
    return  torch.tensor([[0, -omega[3], omega[2]],
                          [omega[3], 0, -omega[1]],
                          [-omega[2], omega[1], 0]])

def L(q):
    return torch.tensor([[q[1] -q[2:4]],
                        [q[2:4], q[1]*torch.eye(3) + hat(q[2:4])]])


def R(q):
    return torch.tensor([[q[1] -q[2:4]],
                        [q[2:4], q[1]*torch.eye(3) - hat(q[2:4])]])

def G(q):
    # attitude jacobian
    return L(q)*H

def rho(phi):
    # convert from ϕ to a quaternion
    return (1/torch.sqrt(1 + phi @ phi))*torch.tensor([[1],[phi]])


