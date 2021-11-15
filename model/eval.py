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

def eval_(args, model):
    print("--- Starting Network Evaluation! ---")
    # determine device
    print("--- Checking for CUDA Device... ---")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # import data
    print("--- Loading validation data... ---")
    val_data = np.load('val_dataset.npz')
    val_inputs = val_data["input"]
    val_labels = val_data["labels"]

    # load into PyTorch Dataset and Dataloader
    val_dataset = DynamicsDataset(val_inputs, val_labels)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     collate_fn=DynamicsDataset.collate_fn,
                                                     pin_memory=True,
                                                     num_workers=1)

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
    average_validation_loss = 0
    for batch_idx, (x, y) in enumerate(val_dataloader):
        x, y = x.to(device), y.to(device)

        x = torch.squeeze(x)
        y_pred = model.forward(x.float())
        loss = loss_fcn(y_pred.unsqueeze(0), y.float())
        average_validation_loss += loss.detach().numpy()

        if batch_idx % 100 == 0:
            print("Iter Num: ", batch_idx)
            print("\t", loss)

    average_validation_loss /= len(val_dataloader)

    print("Average Validation Error: ", average_validation_loss)
