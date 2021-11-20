from dataLoader.DataLoader import DataLoader, DynamicsDataset
import numpy as np

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

DL.load_selected_data(all_flights[:-5])
print(DL.get_time_values().shape)
DL.saveData(saveFileName)
