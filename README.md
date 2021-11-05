# dynamics_model_learning
A repo for CMU's 16-715 Advanced Robot Dynamics course project

## Data Loader Instructions
1. Download the processed data [from the dataset online](https://download.ifi.uzh.ch/rpg/NeuroBEM/) and put it in the top level directory of this project (dynamics_model_learning/)
2. Import the DataLoader class (make sure your script is in top directory - dynamics_model_learning):
`from data.DataLoader import DataLoader`
3. Initialize it with the path to the processed data from the dataset:
`DL = DataLoader(path)`
4. Load in some data by doing one of the following:
    - Use pre-selected set of data: `DL.load_easy_data()`
    - Hand-pick the flights you want from flights_info.txt:
      1. Create a list of the flights you want to load (ex: "2021-02-03-13-44-49"):
        `selected_data = ["2021-02-03-13-44-49", "2021-02-03-13-44-49"]`
      2. Tell the data loader to load in the selected data:
        `DL.load_selected_data(selected_data)`
5. Get chunks of the data as with the following functions:
    - `DL.get_column_names()` will return a list of the coumn names (i.e. state and input variables)
    - `DL.get_time_values()` returns [N] size numpy array of the time values for each data point
    - `DL.get_state_data()` returns [N x 23] numpy array of the state data
        NOTE: includes rpm values as well (part of state but also input to system)
    - `DL.get_control_inputs()` returns [N x 4] numpy array of rpm values (in order 1 -> 4 that fits paper's dynamic model, check Fig 3 in the paper for exact numbering scheme)
    - `DL.get_des_rpm_values()` returns [N x 4] numpy array of desired rpm values at every time step (used to solve for rpm_dot = DL.motor_time_constant * (desired_rpms - current_rpms) )
      - from eq 20 on page 7 of paper
    - `DL.get_battery_voltage_data()` returns [N] size numpy array of battery voltages during flights
