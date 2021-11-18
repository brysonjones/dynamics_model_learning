from dataLoader.DataLoader import DataLoader, DynamicsDataset
import numpy as np

data_folder = "processed_data/"


# Selecting your own data:
flights_to_load = ["2021-02-03-13-43-38", "2021-02-03-13-44-49"]
val_data = ["2021-02-03-13-53-04"]
saveFileName = "val_dataset.npz"

# initialize with path to data folder
DL = DataLoader(data_folder)

DL.load_selected_data(flights_to_load)
DL.saveData(saveFileName)
