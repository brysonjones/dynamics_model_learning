import sys

import json
import numpy as np
import torch

sys.path.append("./model")

from arg_parser import *
from dataLoader.DataLoader import DataLoader, DynamicsDataset
from network import *
from train import *
from eval import *
from compare import *

if __name__ == "__main__":

    args = simulation_args()
    DL_train = DataLoader("./processed_data")
    DL_val = DataLoader("./processed_data")
    param_file = open(args.parameters)
    parameters = json.load(param_file)

    input_size = 21  # number of state dimensions # TODO: generalize this for inputs
    output_size = 6

    model = get_model(args, parameters, input_size, output_size)

    flights_file = "flights_info.txt"
    all_flights = []
    with open(flights_file) as f:
        for line in f.readlines():
            all_flights.append(line.split('\"')[1])

    val_dataloader = None

    # perform main function
    if args.mode == "train":
        col_names = pd.read_csv("processed_data/merged_2021-02-03-13-43-38_seg_1.csv").keys().values
        train_data = DL_train.load_selected_data(all_flights[:-5], cols_to_filter=col_names[1:4])
        # train_data = DL_train.load_selected_data("2021-02-05-14-00-56", cols_to_filter=col_names[1:4])
        # train_data = DL_train.load_easy_data()
        train_dataset = DynamicsDataset(X=DL_train.get_state_data(),
                                        Y=DL_train.state_dot_values)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=model.hyperparams['batch_size'],
                                                         shuffle=True,
                                                         collate_fn=DynamicsDataset.collate_fn,
                                                         pin_memory=True,
                                                         num_workers=1)

        val_data = DL_val.load_selected_data(all_flights[-5:], cols_to_filter=col_names[1:4])
        # val_data = DL_val.load_selected_data("2021-02-05-14-00-56", cols_to_filter=col_names[1:4])
        val_dataset = DynamicsDataset(X=DL_val.get_state_data(),
                                      Y=DL_val.state_dot_values)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                         batch_size=1024,
                                                         shuffle=True,
                                                         collate_fn=DynamicsDataset.collate_fn,
                                                         pin_memory=True,
                                                         num_workers=1)

        train_(args, model, model.hyperparams, train_dataloader, val_dataloader=val_dataloader)
    elif args.mode == "eval":
        eval_(args, model)
    elif args.mode == "simulate":
       pass
    elif args.mode == "compare":
       compare_(args, model)
