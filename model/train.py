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
import torch.utils.data
import pandas as pd

import wandb

def train_(args, model, hyperparams, dataloader, val_dataloader=None):

    if args.wandb:
        wandb.init(project="ML_sys-id_v2", entity="schwartz_code")

        wandb.config = {
          "learning_rate": hyperparams["learning_rate"],
          "epochs": hyperparams["num_epochs"],
          "batch_size": len(dataloader),
          "hidden_layers": "15k"
        }

    print("--- Starting Main Training Loop! ---")
    # determine device
    print("--- Checking for CUDA Device... ---")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.empty_cache()

    # set up training parameters
    learning_rate = hyperparams["learning_rate"]
    weight_decay = hyperparams["weight_decay"]
    num_epochs = hyperparams["num_epochs"]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    if os.path.isfile("model_weights.pth"):
        print("Re-loading existing weights!")
        checkpoint = torch.load("model_weights.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ensure model is in train mode so gradients are properly calculated
    model.train()
    # load device to either GPU or CPU depending on hardware
    model.to(device)

    # set up loss function
    loss_fcn = torch.nn.MSELoss()

    # set up GradScaler to improve run speed
    scaler = torch.cuda.amp.GradScaler()

    print("--- Beginning Training! ---")
    for epoch in range(num_epochs):

        model.train()
        average_training_loss = 0

        print("Epoch #", epoch)

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            for p in model.parameters(): p.grad = None

            x = torch.squeeze(x)
            with torch.cuda.amp.autocast():
                y_pred = model.forward(x.float())

                if(y_pred.size() != y.float().size()):
                    loss = loss_fcn(y_pred.unsqueeze(0), y.float())
                else:
                    loss = loss_fcn(y_pred, y.float())

                loss_data = pd.DataFrame(data=[loss.cpu().detach().numpy()],
                                         columns=["loss"])

            # perform backwards pass
            scaler.scale(loss).backward()

            # run optimization step based on backwards pass
            scaler.step(optimizer)

            # update average training loss
            average_training_loss += loss / len(dataloader)

            # update the scale for next iteration
            scaler.update()

            if batch_idx % 100 == 0:
                print("Iter Num: ", batch_idx)
                print("\t", loss)
                with open("loss_data.csv", 'a') as file:
                    loss_data.to_csv(file, header=False)
            if batch_idx % 5000 == 0:
                print("--- Saving weights ---")
                # save weights after each epoch
                model.to("cpu")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "model_weights.pth")
                model.to(device)

        scheduler.step()

        if val_dataloader is None:
            if args.wandb:
                wandb.log({"average loss": average_training_loss})
        else:
            val_loss = 0
            for batch_idx, (x, y) in enumerate(val_dataloader):
                x, y = x.to(device), y.to(device)

                x = torch.squeeze(x)
                with torch.no_grad():
                    y_pred = model.forward(x.float())

                    if(y_pred.size() != y.float().size()):
                        loss = loss_fcn(y_pred.unsqueeze(0), y.float())
                    else:
                        loss = loss_fcn(y_pred, y.float())

                    loss_data = pd.DataFrame(data=[loss.cpu().detach().numpy()],
                                             columns=["loss"])

                val_loss += loss / len(val_dataloader)
            if args.wandb:
                wandb.log({"average loss": val_loss})
            print("\tValid loss:", val_loss)

    print('end')
