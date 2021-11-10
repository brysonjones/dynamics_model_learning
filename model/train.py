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


def train_(args, model):
    print("--- Starting Main Training Loop! ---")
    # determine device
    print("--- Checking for CUDA Device... ---")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # import data
    print("--- Loading training data... ---")
    train_data = np.load('train_dataset.npz')
    train_inputs = train_data["input"]
    train_labels = train_data["labels"]

    # load into PyTorch Dataset and Dataloader
    train_dataset = DynamicsDataset(train_inputs, train_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     collate_fn=DynamicsDataset.collate_fn,
                                                     pin_memory=True,
                                                     num_workers=1)

    # set up training parameters
    learning_rate = 1e-5
    weight_decay = 1e-6
    num_epochs = 50
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.3)

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

        for batch_idx, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            for p in model.parameters(): p.grad = None

            x = torch.squeeze(x)
            with torch.cuda.amp.autocast():
                y_pred = model.forward(x)
                loss = loss_fcn(y_pred.unsqueeze(0), y.float())
                loss_data = pd.DataFrame(data=[loss.detach().numpy()],
                                         columns=["loss"])

            # perform backwards pass
            scaler.scale(loss).backward()

            # run optimization step based on backwards pass
            scaler.step(optimizer)

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
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "model_weights.pth")

        scheduler.step()

    print('end')
