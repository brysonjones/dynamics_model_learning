"""
Double pendulum motion simulation

"""
from __future__ import print_function

import os
import sys
sys.path.append("")
sys.path.append("../dynamics")

from network import *
from dataloader import *

import numpy as np
import torch
import torch.utils.data
import pandas as pd

def eval_(args, model, dataloader):
    print("--- Starting Network Evaluation! ---")
    # determine device
    print("--- Checking for CUDA Device... ---")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if os.path.isfile("model_weights.pth"):
        print("Re-loading existing weights!")
        checkpoint = torch.load("model_weights.pth")
        model.load_state_dict(checkpoint['model_state_dict'])

    # ensure model is in train mode so gradients are properly calculated
    model.eval()
    # load device to either GPU or CPU depending on hardware
    model.to(device)

    # set up loss function
    loss_fcn = torch.nn.MSELoss()

    print("--- Beginning Evaluating! ---")
    average_validation_error = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        x = torch.squeeze(x)
        y_pred = model.forward(x.float())
        loss = loss_fcn(y_pred.unsqueeze(0), y.float())
        average_validation_error += np.sqrt(loss.detach().numpy())

        # if loss > 1:
        #     print("Iter Num: ", batch_idx)
        #     print("\tValidation Error", loss)
        # if batch_idx % 100 == 0:
        #     print("Iter Num: ", batch_idx)
        #     print("\tValidation Error", loss)

    average_validation_error /= len(dataloader)

    print("Average Validation Error: ", average_validation_error)

    return average_validation_error
