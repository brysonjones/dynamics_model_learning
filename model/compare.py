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
from dataLoader.DataLoader import DataLoader

import numpy as np
import torch
import torch.utils.data
import pandas as pd

def compare_(args, model):
    print("--- Starting Network Evaluation! ---")
    # determine device
    print("--- Checking for CUDA Device... ---")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # import data
    print("--- Loading validation data... ---")
    print('Loading data')
    DL = DataLoader('processed_data')
    DL.load_selected_data('2021-02-05-14-00-56')
    DL.saveData('compare_dataset.npz')
    comp_dataset = np.load('compare_dataset.npz')
    comp_inputs = comp_dataset["input"]
    comp_labels = comp_dataset["labels"]

    # load into PyTorch Dataset and Dataloader
    comp_dataset = DynamicsDataset(comp_inputs, comp_labels)

    # val_dataloader = torch.utils.data.DataLoader(comp_dataset,
    #                                              batch_size=1,
    #                                              shuffle=False,
    #                                              collate_fn=DynamicsDataset.collate_fn,
    #                                              pin_memory=True,
    #                                              num_workers=1)

    if os.path.isfile("model_weights.pth"):
        print("Re-loading existing weights!")
        checkpoint = torch.load("model_weights.pth")
        model.load_state_dict(checkpoint['model_state_dict'])

    # ensure model is in train mode so gradients are properly calculated
    model.eval()
    # load device to either GPU or CPU depending on hardware
    model.to(device)

    x = torch.tensor(comp_inputs)
    x = torch.squeeze(x)
    y_pred = model.forward(x.float())

    print("lmao it worked :o")
    print(y_pred.size)