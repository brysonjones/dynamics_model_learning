import sys

sys.path.append("./model")

from arg_parser import *
from network import *
from train import *
from eval import *

if __name__ == "__main__":

    args = simulation_args()

    input_size = 12  # number of state dimensions # TODO: generalize this for inputs
    output_size = 1  # for all lagrangian systems, output should be just a scalar energy value
    model = get_model(args, input_size, output_size)
    
    # perform main function
    if args.mode == "train":
        train_(args, model)
    elif args.mode == "eval":
        eval_(args, model)
    elif args.mode == "simulate":
       pass


