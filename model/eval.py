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
from dynamics.manipulator import *

import numpy as np
import torch
import torch.utils.data
from torch.nn.functional import mse_loss as mse_fcn
import pandas as pd

def eval_(model, dataloader):
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
    x_error = 0
    y_error = 0
    z_error = 0
    qw_error = 0
    qx_error = 0
    qy_error = 0
    qz_error = 0

    average_trans_validation_error = 0
    average_quat_validation_error = 0
    average_validation_error = 0

    val_data = pd.DataFrame(data=["X", "Y", "Z", "qw", "qx", "qy", "qz"]).T
    with open("error_data.csv", 'w') as file:
        val_data.to_csv(file, header=False)

    for batch_idx, (q1, q2, q3, u1, u2, u3) in enumerate(dataloader):
        q1, q2, q3, u1, u2, u3 = torch.squeeze(q1.to(device)), \
                                 torch.squeeze(q2.to(device)), \
                                 torch.squeeze(q3.to(device)), \
                                 torch.squeeze(u1.to(device)), \
                                 torch.squeeze(u2.to(device)), \
                                 torch.squeeze(u3.to(device))

        q3_pred = model.step(q1, q2, u1, u2, u3)
        loss = loss_fcn(q3_pred, q3.float())

        x_error = np.abs((q3_pred[0] - q3[0].float()).detach().numpy())
        y_error = np.abs((q3_pred[1] - q3[1].float()).detach().numpy())
        z_error = np.abs((q3_pred[2] - q3[2].float()).detach().numpy())
        qw_error = np.abs((q3_pred[3] - q3[3].float()).detach().numpy())
        qx_error = np.abs((q3_pred[4] - q3[4].float()).detach().numpy())
        qy_error = np.abs((q3_pred[5] - q3[5].float()).detach().numpy())
        qz_error = np.abs((q3_pred[6] - q3[6].float()).detach().numpy())

        val_data = pd.DataFrame(data=[x_error, y_error, z_error, qw_error, qx_error, qy_error, qz_error]).T
        with open("error_data.csv", 'a') as file:
            val_data.to_csv(file, header=False)

        average_trans_validation_error +=mse_fcn(q3_pred[0:3], q3[0:3].float()).detach().numpy()
        average_quat_validation_error += mse_fcn(q3_pred[3:7], q3[3:7].float()).detach().numpy()
        average_validation_error += loss.detach().numpy()

        if loss > 1:
            print("Iter Num: ", batch_idx)
            print("\tValidation Error", loss)
        if batch_idx % 20  == 0:
            print("Iter Num: ", batch_idx)
            print("\tValidation Error", loss)

    average_trans_validation_error /= len(dataloader)
    average_quat_validation_error /= len(dataloader)
    average_validation_error /= len(dataloader)

    print("Average Translation Validation Error: ", average_trans_validation_error)
    print("Average Quaternion Validation Error: ", average_quat_validation_error)
    print("Average Validation Error: ", average_validation_error)

    return average_trans_validation_error, average_quat_validation_error, average_validation_error

def eval_accel(model, dataloader):
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
    average_trans_validation_error = 0
    average_quat_validation_error = 0
    average_validation_error = 0
    for batch_idx, (qq, q_dotdot) in enumerate(dataloader):
        qq, q_dotdot = torch.squeeze(qq.to(device)), torch.squeeze(q_dotdot.to(device))
        q = qq[[0, 1, 2, 9, 6, 7, 8]]
        q_dot = qq[[3, 4, 5, 10, 11, 12]]
        u = qq[13:17]
        q_dotdot_pred = calc_accel(model, q, q_dot, u)
        loss = loss_fcn(q_dotdot_pred, q_dotdot.float())

        average_trans_validation_error += np.sqrt(mse_fcn(q_dotdot_pred[0:3], q_dotdot[0:3].float()).detach().numpy())
        average_quat_validation_error += np.sqrt(mse_fcn(q_dotdot_pred[3:6], q_dotdot_pred[3:6].float()).detach().numpy())
        average_validation_error += np.sqrt(loss.detach().numpy())

        if loss > 1:
            print("Iter Num: ", batch_idx)
            print("\tValidation Error", loss)
        if batch_idx % 20 == 0:
            print("Iter Num: ", batch_idx)
            print("\tValidation Error", loss)

    average_trans_validation_error /= len(dataloader)
    average_quat_validation_error /= len(dataloader)
    average_validation_error /= len(dataloader)

    print("Average Translation Validation Error: ", average_trans_validation_error)
    print("Average Quaternion Validation Error: ", average_quat_validation_error)
    print("Average Validation Error: ", average_validation_error)

    return average_trans_validation_error, average_quat_validation_error, average_validation_error
