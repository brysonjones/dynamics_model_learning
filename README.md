# dynamics_model_learning
A repo for CMU's 16-715 Advanced Robot Dynamics course project

## Data Loader Instructions
  1. Download the processed data [from the dataset online](https://download.ifi.uzh.ch/rpg/NeuroBEM/) and put it in the top level directory of this project (dynamics_model_learning/)
  2. Import the DataLoader class (make sure your script is in top directory - dynamics_model_learning):
  `from data.DataLoader import DataLoader, DynamicsDataset`
  3. Initialize it with the path to the processed data from the dataset:
  `DL = DataLoader(path)`
  4. Load in some data by doing one of the following:
    - Use pre-selected set of data: `DL.load_easy_data()`
    - Hand-pick the flights you want from flights_info.txt:
      1. Create a list of the flights you want to load (ex: "2021-02-03-13-44-49"):
        `selected_data = ["2021-02-03-13-44-49", "2021-02-03-13-44-49"]`
      2. Tell the data loader to load in the selected data:
        `DL.load_selected_data(selected_data)`
  6. To save the data to an npz file, do: `DL.saveData(path)`, where path is the path to the file you want the data to be saved in (ex: model/train_data.npz)
    - The data can be loaded into a file using: `npzfile = np.load(filePath)`
    - The state data can be accessed with: `npzfile["input"]`
    - The control inputs (RPM values) can be accessed with: `npzfile["control_inputs"]`
    - The state derivative can be accessed with: ``
  7. You can now train a network using this data, see dynamics_model_learning/model/train.py for an example on how to do that.
  8. (optional) Load the data into a torch.utils.data.Dataset function by doing:
    `torchDatasetObject = DynamicsDataset(DL.get_state_data(),  DL.get_control_inputs())`
  9. (optional) Get chunks of the data as with the following functions:
      - `DL.get_column_names()` will return a list of the coumn names (i.e. state and input variables)
      - `DL.get_time_values()` returns [N] size numpy array of the time values for each data point
      - `DL.get_state_data()` returns [N x 23] numpy array of the state data
          NOTE: includes rpm values as well (part of state but also input to system)
      - `DL.get_control_inputs()` returns [N x 4] numpy array of rpm values (in order 1 -> 4 that fits paper's dynamic model, check Fig 3 in the paper for exact numbering scheme)
      - `DL.get_des_rpm_values()` returns [N x 4] numpy array of desired rpm values at every time step (used to solve for rpm_dot = DL.motor_time_constant * (desired_rpms - current_rpms) )
        - from eq 20 on page 7 of paper
      - `DL.get_battery_voltage_data()` returns [N] size numpy array of battery voltages during flights
