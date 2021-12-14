"""
Double pendulum motion simulation

"""
from __future__ import print_function

import os
import sys
sys.path.append("")
sys.path.append("../dynamics")
sys.path.append('./Visualizers')

from network import *
from dataloader import *
from dataLoader.DataLoader import DataLoader
from visTool import accCompare

import numpy as np
import torch
import torch.utils.data
import pandas as pd
import wandb

def compare_(args, model, wandb=False):
    print("--- Starting Network Evaluation! ---")
    # determine device
    print("--- Checking for CUDA Device... ---")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # load in all data
    flights_file = "flights_info.txt"

    all_flights = []
    with open(flights_file) as f:
        for line in f.readlines():
            all_flights.append(line.split('\"')[1])

    # import data
    print("--- Loading validation data... ---")
    print('Loading data')
    DL = DataLoader('processed_data')
    col_names = pd.read_csv("processed_data/merged_2021-02-03-13-43-38_seg_1.csv").keys().values
    print(col_names)
    # DL.load_selected_data('2021-02-05-14-00-56', cols_to_filter=col_names[1:4]) # from train set
    # DL.load_selected_data('2021-02-03-17-14-47', cols_to_filter=col_names[1:4])
    DL.load_selected_data("2021-02-23-22-54-17", cols_to_filter=col_names[1:4]) # from val set
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

    x = torch.tensor(comp_inputs).to(device)
    x = torch.squeeze(x)
    predAcc = model.forward(x.float()).cpu().detach().numpy()

    actualAcc = DL.state_dot_values

    timeVec = DL.get_time_values()

    accCompare(timeVec, actualAcc, predAcc)

    avg_MSE_loss = np.sum(np.sqrt( (actualAcc - predAcc)**2 )) / len(timeVec)
    print("Average MSE per time step: ", avg_MSE_loss)
