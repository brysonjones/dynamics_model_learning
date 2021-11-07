
import torch
import sys
sys.path.append("../dynamics")

from lagrangian import *

def get_model(args, D_in, D_out):

    print("--- Constructing Model... ---")
    # TODO: parameterize model construction

    model = None
    if args.model == "LNN":
        hidden_list = [D_in, 512, 512, 512, 512, 512, 512]
        model = LagrangianNeuralNetwork(D_in, hidden_list, D_out)
    elif args.model == "BlackBox":
        pass
    elif args.model == "NewtonEuler":
        pass
    else:
        print("No Model Provided -- Please try again")
        raise Exception

    return model


class LagrangianNeuralNetwork(torch.nn.Module):
    def __init__(self, D_in, hidden_list, D_out):
        """
        Neural Network used to approximate a paramaterized system lagrangian
        """
        super(LagrangianNeuralNetwork, self).__init__()
        self.model_layers = torch.nn.ModuleList()

        # input layer
        self.model_layers.append(torch.nn.Linear(D_in, hidden_list[0]))
        self.model_layers.append(torch.nn.Softplus())
        # add all hiden layers
        for i in range(1, len(hidden_list)):
            self.model_layers.append(torch.nn.Linear(hidden_list[i-1], hidden_list[i]))
            self.model_layers.append(torch.nn.Softplus())

        # output layer
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], D_out))
        self.model_layers.append(torch.nn.Softplus())

    def forward_nn(self, x):
        """
        applies all of the model layers, and returns the single output value,
        which in this case is the lagrangian of the system, representing the
        total energy
        """
        for layer in self.model_layers:
            x = layer(x)

        return x

    def forward(self, x):
        """
        solves for the generalized acceleration using the forward_nn() fcn
        to calculate the Lagrangian
        """
        x_dot = solve_euler_lagrange(self.forward_nn, x.float())

        return x_dot


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, D_in, hidden_list, D_out):
        """
        Neural Network used to approximate a paramaterized system lagrangian
        """
        super(FullyConnectedNetwork, self).__init__()
        self.model_layers = torch.nn.ModuleList()

        # input layer
        self.model_layers.append(torch.nn.Linear(D_in, hidden_list[0]))
        self.model_layers.append(torch.nn.BatchNorm1d(hidden_list[0]))
        self.model_layers.append(torch.nn.Softplus())
        # add all hiden layers
        for i in range(1, len(hidden_list)):
            self.model_layers.append(torch.nn.Linear(hidden_list[i-1], hidden_list[i]))
            self.model_layers.append(torch.nn.BatchNorm1d(hidden_list[i]))
            self.model_layers.append(torch.nn.Softplus())

        # output layer
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], D_out))
        self.model_layers.append(torch.nn.Softplus())

    def forward(self, x):
        """
        applies all of the model layers, and returns the single output value,
        which in this case is the lagrangian of the system, representing the
        total energy
        """
        tmp = x
        for layer in self.model_layers:
            tmp = layer(tmp)
        out = tmp

        return out

