import sys

import json

import torch

sys.path.append("./model")

from arg_parser import *
from model.network import *
from model.DEL_network import *
from model.train import *
from model.eval import *
from dynamics.simulate import *
from dataLoader.DataLoader import DataLoader, DynamicsDataset, DELDataset, DELTestDataset
from torch.nn.functional import mse_loss as mse_fcn

if __name__ == "__main__":

    args = simulation_args()
    DL = DataLoader("./processed_data")
    param_file = open(args.parameters)
    parameters = json.load(param_file)

    input_size = 17  # number of state dimensions for dataset
    num_pose_states = 7
    num_control_inputs = 4
    output_size = 6

    model = get_model(args, parameters, input_size, output_size)
    # get dataset
    col_names = pd.read_csv("processed_data/merged_2021-02-03-13-43-38_seg_1.csv").keys().values
    DL.load_selected_data(DL.short_circles[6], cols_to_filter=col_names[0:8])
    train_data_1 = DL.get_state_data()
    np.save("train.npy", train_data_1)
    DL.load_selected_data(DL.linear_oscillations[0], cols_to_filter=col_names[0:8])
    train_data_2 = DL.get_state_data()
    DL.load_selected_data(DL.vertical_oscillations[0], cols_to_filter=col_names[0:8])
    train_data_3 = DL.get_state_data()

    all_train_data = [train_data_1, train_data_2, train_data_3]

    np.savez("train.npz", data_1=train_data_1, data_2=train_data_2, data_3=train_data_3)

    # DL.load_selected_data(DL.linear_oscillations[0], cols_to_filter=col_names[0:8])
    DL.load_selected_data(DL.short_circles[0], cols_to_filter=col_names[0:8])
    eval_data = DL.get_state_data()[:5000, :]
    np.save("eval.npy", eval_data)

    DL.load_selected_data(DL.short_circles[6], cols_to_filter=col_names[0:8])
    test_data = DL.get_state_data()[:, [0, 1, 2, 9, 6, 7, 8, 13, 14, 15, 16]]
    np.save("test.npy", test_data)

    if args.model == "DELN":
        train_dataset = DELDataset(X=all_train_data)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=model.hyperparams['batch_size'],
                                                       shuffle=True,
                                                       collate_fn=DELDataset.collate_fn,
                                                       pin_memory=True,
                                                       num_workers=1)
        eval_dataset = DELTestDataset(X=eval_data)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=model.hyperparams['batch_size'],
                                                      shuffle=False,
                                                      collate_fn=DELTestDataset.collate_fn,
                                                      pin_memory=True,
                                                      num_workers=1)
        test_dataset = DELTestDataset(X=test_data)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=model.hyperparams['batch_size'],
                                                      shuffle=False,
                                                      collate_fn=DELTestDataset.collate_fn,
                                                      pin_memory=True,
                                                      num_workers=1)
    else:
        DL.load_selected_data(DL.short_circles[1])
        train_dataset = DynamicsDataset(X=DL.get_state_data(),
                                        Y=DL.state_dot_values)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=model.hyperparams['batch_size'],
                                                       shuffle=True,
                                                       collate_fn=DynamicsDataset.collate_fn,
                                                       pin_memory=True,
                                                       num_workers=1)
        DL.load_selected_data(DL.short_circles[0])
        eval_dataset = DynamicsDataset(X=DL.get_state_data(),
                                       Y=DL.state_dot_values)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=model.hyperparams['batch_size'],
                                                      shuffle=False,
                                                      collate_fn=DynamicsDataset.collate_fn,
                                                      pin_memory=True,
                                                      num_workers=1)
    val_dataloader = None

    # perform main function
    if args.mode == "train":
        train_(args, model, model.hyperparams, train_dataloader, eval_dataloader)
    elif args.mode == "eval":
        eval_(model, eval_dataloader)
    elif args.mode == "test":
        eval_(model, test_dataloader)
    elif args.mode == "test_accel":
        test_dataloader = DynamicsDataset(X=DL.get_state_data(),
                                          Y=DL.state_dot_values)
        eval_accel(model, test_dataloader)
    elif args.mode == "simulate":
        error = np.zeros(test_data.shape[0])
        q_pred = torch.zeros((test_data.shape[0], 7))
        test_data = torch.as_tensor(test_data)
        q_pred[0, :] = test_data[0, 0:7]
        q_pred[1, :] = test_data[1, 0:7]
        u_true = test_data[:, 7:]
        for i in range(2, test_data.shape[0]):
            q_pred[i, :] = model.step(q_pred[i-2, :].float(), q_pred[i-1, :].float(), u_true[i-2, :].float(), u_true[i-1, :].float(), u_true[i, :].float())
            error[i] = np.sqrt(mse_fcn(q_pred[i, :], test_data[i, 0:7].float()).detach().numpy())
            print(error[i])
