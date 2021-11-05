import os
import pandas as pd
import math as m
import numpy as np

class DataLoader(object):

    def __init__(self, path_to_data):
        self.data_path = path_to_data
        self.data = None #stores pandas dataframes
        self.selected_data = None #just filenames to be loaded in (see self.load_selected_data())

        self.state_columns = ['ang acc x', 'ang acc y', 'ang acc z', 'ang vel x', 'ang vel y',
                            'ang vel z', 'quat x', 'quat y', 'quat z', 'quat w', 'acc x', 'acc y',
                            'acc z', 'vel x', 'vel y', 'vel z', 'pos x', 'pos y', 'pos z', 'mot 1',
                            'mot 2', 'mot 3', 'mot 4']

        self.motor_speed_columns = ['mot 1', 'mot 2', 'mot 3', 'mot 4']
        self.motor_derivative_columns = ['dmot 1', 'dmot 2', 'dmot 3', 'dmot 4']
        self.motor_time_constant = 1 / 0.033 # motor constant k where rpm_dot = k*(rpm_des - rpm_curr)
        # this is from page 7 of the NeuroBEM paper

        # types of flights from labels online
        self.short_circles = ['2021-02-05-14-00-56', '2021-02-05-14-01-47', '2021-02-05-14-02-47', '2021-02-05-14-03-41', '2021-02-05-14-04-32',
                              '2021-02-05-16-16-00', '2021-02-05-16-19-10']

        # oscillating in z
        self.vertical_oscillations = ['2021-02-05-14-19-34', '2021-02-05-14-24-35']

        # oscillating in x and/or y
        self.linear_oscillations = ["2021-02-03-16-10-37", "2021-02-03-16-12-22", "2021-02-03-16-45-38", "2021-02-03-16-54-28", "2021-02-18-16-41-41",
                                   "2021-02-18-16-43-54", "2021-02-18-16-47-02", "2021-02-18-16-48-24", "2021-02-18-16-53-35", "2021-02-18-16-55-00"]

    def load_easy_data(self):
        self.data = None

        easy_data = self.short_circles + self.vertical_oscillations + self.linear_oscillations

        flights = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]

        for f in flights:
            if f[7:-10] in easy_data:
                if self.data is None:
                    self.data = pd.read_csv(os.path.join(self.data_path, f))
                else:
                    self.data = pd.concat([self.data, pd.read_csv(os.path.join(self.data_path, f))])


    def get_column_names(self):
        return self.data.keys()


    def load_selected_data(self):
        if self.selected_data is None:
            print("ERROR: you must set DataLoader.selected_data equal to a list of the files \
                   you want to load data from before calling this function!")
            return None

        self.data = None
        flights = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]

        if len(flights) == 0:
            print("ERROR! No csv files found with the names provided: \n", self.selected_data)

        for f in flights:
            if f[7:-10] in self.selected_data:
                if self.data is None:
                    self.data = pd.read_csv(os.path.join(self.data_path, f))
                else:
                    self.data = pd.concat([self.data, pd.read_csv(os.path.join(self.data_path, f))])

    def get_time_values(self):
        return self.data['t'].values

    def get_state_data(self):
        # indexes columns
        return self.data[self.state_columns].values

    def get_control_inputs(self):
        return self.data[self.motor_speed_columns].values

    def get_des_rpm_values(self):
        dt = 0.001 # [sec] (1 kHz)
        rpm_dot_vals = (np.diff(self.data[self.motor_speed_columns].values, axis=0)) / dt
        # copy last time step so rpm_dot_vals is same length as data
        rpm_dot_vals = np.vstack([rpm_dot_vals, rpm_dot_vals[-1,:]])

        return rpm_dot_vals / self.motor_time_constant + self.data[self.motor_speed_columns].values


if __name__ == '__main__':
    DL = DataLoader("processed_data/")
    DL.load_easy_data()
    print(DL.get_column_names())
    # test = DL.get_state_data()
    # print(test.shape)
    print(DL.get_des_rpm_values().shape)
    print("testing ==================")

    a = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    print(a.shape)
    print(np.diff(a, axis=0))
