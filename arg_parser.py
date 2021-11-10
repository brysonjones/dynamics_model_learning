
import argparse


def simulation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parameters",
                        help="path to json file specifying simulation parameters")
    args = parser.parse_args()

    return args
