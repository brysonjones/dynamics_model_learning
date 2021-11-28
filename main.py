import sys

import json

sys.path.append("./model")

from arg_parser import *
from dataLoader.DataLoader import DataLoader, DynamicsDataset, DELDataset
from model.network import *
from model.DEL_network import *
from model.train import *
from model.eval import *

if __name__ == "__main__":

    args = simulation_args()
    DL = DataLoader("./processed_data")
    param_file = open(args.parameters)
    parameters = json.load(param_file)

    input_size = 17  # number of state dimensions # TODO: generalize this for inputs
    output_size = 6

    model = get_model(args, parameters, input_size, output_size)
    # get dataset
    train_data = DL.load_easy_data()
    if args.model == "DELN":
        train_dataset = DELDataset(X=DL.get_state_data())
    else:
        train_dataset = DynamicsDataset(X=DL.get_state_data(),
                                        Y=DL.state_dot_values)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=model.hyperparams['batch_size'],
                                                     shuffle=True,
                                                     collate_fn=DynamicsDataset.collate_fn,
                                                     pin_memory=True,
                                                     num_workers=1)
    val_dataloader = None

    # perform main function
    if args.mode == "train":
        train_(args, model, model.hyperparams, train_dataloader)
    elif args.mode == "eval":
        eval_(args, model, val_dataloader)
    elif args.mode == "simulate":
       pass
