from dataLoader.DataLoader import DataLoader, DynamicsDataset

DL = DataLoader("processed_data/")

# Using pre-selected data:
# DL.load_easy_data()

# Selecting your own data:
flights_to_load = ["2021-02-03-13-43-38", "2021-02-03-13-44-49"]

# Member functions of DataLoader class:
DL.load_selected_data(flights_to_load)
print("Column names:\t", DL.get_column_names())
print("Time value array shape:\t", DL.get_time_values().shape)
print("State data (including rpm values) array shape:\t", DL.get_state_data().shape)
print("Control inputs (rpm values) array shape:\t", DL.get_control_inputs().shape)
print("Desired rpm values array shape:\t\t", DL.get_des_rpm_values().shape)
print("State dot array size: ", DL.state_dot_values.shape)

DL.saveData("simple_dataset.npz")

tmp = np.load("test.npz")
inputs = tmp["input"]
outputs = tmp["output"]

test = DynamicsDataset(DL.get_state_data(), DL.state_dot_values)
print(test[0])
print(len(test))
