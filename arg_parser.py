
import argparse


def simulation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parameters",
                        help="path to json file specifying simulation parameters")
    parser.add_argument("-m", "--model",
                        help="type of model you want to train")
    parser.add_argument("-n", "--mode",
                        help="which step in training process you want (train/eval/simulate)")
    parser.add_argument("-w", "--wandb", default=False,
                        help="set to true if you want to use wandb to track loss")
    args = parser.parse_args()

    return args
