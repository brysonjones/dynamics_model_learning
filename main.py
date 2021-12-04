import sys

import json

sys.path.append("./model")

from arg_parser import *
from dataLoader.DataLoader import DataLoader, DynamicsDataset, DELDataset
from model.network import *
from model.DEL_network import *
from model.train import *
from model.eval import *
from dynamics.simulate import *

if __name__ == "__main__":

    args = simulation_args()
    DL = DataLoader("./processed_data")
    param_file = open(args.parameters)
    parameters = json.load(param_file)

    input_size = 17  # number of state dimensions for dataset
    output_size = 6

    model = get_model(args, parameters, input_size, output_size)
    # get dataset
    # train_data = DL.load_easy_data()
    DL.load_selected_data(DL.short_circles[1])
    train_data = DL.get_state_data()
    np.save("train.npy", train_data)
    DL.load_selected_data(DL.short_circles[0])
    eval_data = DL.get_state_data()
    np.save("eval.npy", eval_data)
    if args.model == "DELN":
        train_dataset = DELDataset(X=train_data)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=model.hyperparams['batch_size'],
                                                       shuffle=True,
                                                       collate_fn=DELDataset.collate_fn,
                                                       pin_memory=True,
                                                       num_workers=1)
        eval_dataset = DELDataset(X=eval_data)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=model.hyperparams['batch_size'],
                                                      shuffle=False,
                                                      collate_fn=DELDataset.collate_fn,
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
        eval_(args, model, eval_dataloader)
    elif args.mode == "simulate":
        easy_data = DL.get_state_data()

        q1 = easy_data[0, [0, 1, 2, 9, 6, 7, 8]]
        q2 = easy_data[1, [0, 1, 2, 9, 6, 7, 8]]
        u_array = easy_data[:, 13:17]
        # simulate_(model, q1, q2, u_array)
