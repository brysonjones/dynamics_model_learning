
import torch
import sys
sys.path.append("../dynamics")

from dynamics.lagrangian import *

def get_model(args, parameters, D_in, D_out):

    print("--- Constructing Model... ---")
    # TODO: parameterize model construction

    model = None
    if args.model == "LNN":
        hidden_list = [512, 512, 512, 512, 512, 512]
        control_hidden = [128, 128, 128, 128, 128, 128]
        model = LagrangianNeuralNetwork(D_in, hidden_list, control_hidden, D_out)
    elif args.model == "FCN":
        hidden_list = [17, 512, 512, 512, 512, 512, 512]
        model = FullyConnectedNetwork(D_in, hidden_list, D_out)
    elif args.model == "NewtonEuler":
        pass
    else:
        print("NULL/Incorrect Model Choice Provided -- Please try again")
        raise Exception

    model.hyperparams = parameters['model'][args.model]['hyperparameters']

    return model


class LagrangianNeuralNetwork(torch.nn.Module):
    def __init__(self, D_in, lagrange_hidden, control_hidden, D_out):
        """
        Neural Network used to approximate a paramaterized system lagrangian
        """
        super(LagrangianNeuralNetwork, self).__init__()
        self.hyperparams = None
        self.lagrange_model_layers = torch.nn.ModuleList()
        self.external_force_layers = torch.nn.ModuleList()

        # input layer - Lagrange
        self.lagrange_model_layers.append(torch.nn.Linear(D_in, lagrange_hidden[0]))
        self.lagrange_model_layers.append(torch.nn.Softplus())
        # add all hidden layers
        for i in range(0, len(lagrange_hidden)-1):
            self.lagrange_model_layers.append(torch.nn.Linear(lagrange_hidden[i], lagrange_hidden[i+1]))
            self.lagrange_model_layers.append(torch.nn.Softplus())

        # output layer
        # output is always one from this network because it is calculating system energy
        self.lagrange_model_layers.append(torch.nn.Linear(lagrange_hidden[-1], D_out))
        self.lagrange_model_layers.append(torch.nn.Softplus())

        ### Control Force Network ###
        # input layer - Lagrange
        self.external_force_layers.append(torch.nn.Linear(D_in, control_hidden[0]))
        self.external_force_layers.append(torch.nn.LeakyReLU(.025))
        # add all hidden layers
        for i in range(0, len(control_hidden) - 1):
            self.external_force_layers.append(torch.nn.Linear(control_hidden[i], control_hidden[i + 1]))
            self.external_force_layers.append(torch.nn.LeakyReLU(.025))

        # output layer
        control_output_dofs = 6  # TODO: remove this hardcode
        self.external_force_layers.append(torch.nn.Linear(control_hidden[-1], control_output_dofs))

    def forward_lagrange(self, x):
        """
        applies all of the model layers, and returns the single output value,
        which in this case is the lagrangian of the system, representing the
        total energy
        """
        for layer in self.lagrange_model_layers:
            x = layer(x)

        return x


    def forward_external_force(self, x):
        """
        applies all of the model layers, and returns the estimated force and torque
        values applied to the system
        """
        for layer in self.external_force_layers:
            x = layer(x)

        return x

    def forward(self, x):
        """
        solves for the generalized acceleration using the forward_nn() fcn
        to calculate the Lagrangian
        """
        xx = x[1, 0:8]
        xxt = x[8, 14]
        ext_input = x[14:-1]
        x_dot = solve_euler_lagrange(self.forward_lagrange, self.forward_external_force,
                                     xx.float(), xxt.float(), ext_input.float())

        return x_dot


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, D_in, hidden_list, D_out):
        """
        Neural Network used to approximate a paramaterized system lagrangian
        """
        super(FullyConnectedNetwork, self).__init__()
        self.hyperparams = None
        self.model_layers = torch.nn.ModuleList()

        # input layer
        self.model_layers.append(torch.nn.Linear(D_in, hidden_list[0]))
        self.model_layers.append(torch.nn.LeakyReLU())
        # add all hidden layers
        for i in range(1, len(hidden_list)):
            self.model_layers.append(torch.nn.Linear(hidden_list[i-1], hidden_list[i]))
            self.model_layers.append(torch.nn.LeakyReLU())

        # output layer
        self.model_layers.append(torch.nn.Linear(hidden_list[-1], D_out))

    def forward(self, x):
        """
        applies all of the model layers
        """
        tmp = x
        for layer in self.model_layers:
            tmp = layer(tmp)
        out = tmp

        return out
