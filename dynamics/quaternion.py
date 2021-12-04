
import torch

H = torch.cat((torch.zeros(1,3), torch.eye(3)), dim=0)

# Some standard functions for dealing with rotation matrices and quaternions


def hat(omega):
    return  torch.tensor([[0, -omega[2], omega[1]],
                          [omega[2], 0, -omega[0]],
                          [-omega[1], omega[0], 0]])


def L(q):
    return torch.vstack((torch.hstack((q[0], -q[1:4])),
                         torch.hstack((torch.unsqueeze(q[1:4], 1), q[0]*torch.eye(3) + hat(q[1:4])))))


def R(q):
    return torch.vstack((torch.hstack((q[0], -q[1:4])),
                        torch.hstack((q[1:4], q[0]*torch.eye(3) - hat(q[1:4])))))


def G(q):
    # attitude jacobian
    return L(q)@H


def G_(q):
    Q = q[3:7]
    return torch.vstack((torch.hstack((torch.eye(3), torch.zeros(3,3))),
                         torch.hstack((torch.zeros(4,3), G(Q)))))

def rho(phi):
    # convert from ϕ to a quaternion
    return (1/torch.sqrt(1 + phi @ phi))*torch.hstack((torch.tensor(1), phi))


