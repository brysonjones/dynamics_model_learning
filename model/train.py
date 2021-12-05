
import sys
sys.path.append("")
sys.path.append("../dynamics")

from eval import *

import torch.utils.data
import pandas as pd


def train_(args, model, hyperparams, train_dataloader, eval_dataloader):
    print("--- Starting Main Training Loop! ---")
    # determine device
    print("--- Checking for CUDA Device... ---")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # set up training parameters
    learning_rate = hyperparams["learning_rate"]
    weight_decay = hyperparams["weight_decay"]
    num_epochs = hyperparams["num_epochs"]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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

    # set up csv files for logging
    loss_data = pd.DataFrame(data=["Epoch Num", "Average Loss"]).T
    with open("training_loss_data.csv", 'w') as file:
        loss_data.to_csv(file, header=False)
    val_data = pd.DataFrame(data=["Epoch Num", "Average Error"]).T
    with open("val_loss_data.csv", 'w') as file:
        val_data.to_csv(file, header=False)

    print("--- Beginning Training! ---")
    for epoch in range(num_epochs):

        model.train()

        print("Epoch #", epoch)
        average_del_res = 0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            for p in model.parameters(): p.grad = None

            x = torch.squeeze(x)
            with torch.cuda.amp.autocast():
                y_pred = model.forward(x.float())
                loss = loss_fcn(y_pred.unsqueeze(0), y.float())
                loss_data = pd.DataFrame(data=[loss.detach().numpy()],
                                         columns=["loss"])
                average_del_res += loss.detach().numpy()

            # perform backwards pass
            scaler.scale(loss).backward()

            # clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

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

        average_del_res /= len(train_dataloader)
        print("Average DEL Resiual over Epoch: ", average_del_res)
        loss_data = pd.DataFrame(data=[epoch, average_del_res]).T
        with open("training_loss_data.csv", 'a') as file:
            loss_data.to_csv(file, header=False)

        avg_val_error = eval_(model, eval_dataloader)
        val_data = pd.DataFrame(data=[epoch, avg_val_error]).T
        with open("val_loss_data.csv", 'a') as file:
            val_data.to_csv(file, header=False)

        scheduler.step()

    print('end')
