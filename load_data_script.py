from dataLoader.DataLoader import DataLoader, DynamicsDataset
import numpy as np
import pandas as pd
import os

data_folder = "processed_data/"
flights_file = "flights_info.txt"

all_flights = []
with open(flights_file) as f:
    for line in f.readlines():
        all_flights.append(line.split('\"')[1])

# Selecting your own data:
flights_to_load = ["2021-02-03-13-43-38", "2021-02-03-13-44-49"]
val_data = ["2021-02-03-13-53-04"]
saveFileName = "train_dataset.npz"

# initialize with path to data folder
DL = DataLoader(data_folder)

# DL.load_selected_data(val_data)
col_names = pd.read_csv("processed_data/merged_2021-02-03-13-43-38_seg_1.csv").keys().values
print(col_names)
DL.load_selected_data(all_flights[:-5], cols_to_filter=col_names[1:4])
# DL.saveData(saveFileName)
