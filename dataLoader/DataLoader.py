import os
import pandas as pd
import math as m
import numpy as np
import torch
import torch.utils.data
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DataLoader(object):

    def __init__(self, path_to_data):
        self.data_path = path_to_data
        self.data = None #stores pandas dataframes
        self.selected_data = None #just filenames to be loaded in (see self.load_selected_data())
        self.length = 0

        self.state_columns = None
        self.motor_speed_columns = None
        self.motor_derivative_columns = None

        self.motor_time_constant = 1 / 0.033 # motor constant k where rpm_dot = k*(rpm_des - rpm_curr)
        # this is from page 7 of the NeuroBEM paper

        self.N = 50 # number of adjacent data points averaged in moving average filter

        self.state_columns = ['pos x', 'pos y', 'pos z', 'vel x', 'vel y', 'vel z', 'quat x',
                              'quat y', 'quat z', 'quat w', 'ang vel x', 'ang vel y',
                              'ang vel z', 'mot 1', 'mot 2', 'mot 3', 'mot 4']

        # types of flights from labels online
        self.short_circles = ['2021-02-05-14-00-56', '2021-02-05-14-01-47', '2021-02-05-14-02-47', '2021-02-05-14-03-41', '2021-02-05-14-04-32',
                              '2021-02-05-16-16-00', '2021-02-05-16-19-10']

        # oscillating in z
        self.vertical_oscillations = ['2021-02-05-14-19-34', '2021-02-05-14-24-35']

        # oscillating in x and/or y
        self.linear_oscillations = ["2021-02-03-16-10-37", "2021-02-03-16-12-22", "2021-02-03-16-45-38", "2021-02-03-16-54-28", "2021-02-18-16-41-41",
                                   "2021-02-18-16-43-54", "2021-02-18-16-47-02", "2021-02-18-16-48-24", "2021-02-18-16-53-35", "2021-02-18-16-55-00"]

        self.dt = 1/400 # [sec] (400 Hz, from paper)

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

        self.motor_speed_columns = self.get_column_names()[-9:-5]
        self.motor_derivative_columns = self.get_column_names()[-5:-1]

        self.length = self.data[self.state_columns].values.shape[0]

        self.calculate_state_dot_values()

    def get_column_names(self):
        return self.data.keys().values

    def load_selected_data(self, selected_data, cols_to_filter=None):
        self.selected_data = selected_data

        self.data = None
        flights = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]

        if len(flights) == 0:
            print("ERROR! No csv files found with the names provided: \n", selected_data)

        for f in flights:
            if f[7:-10] in selected_data:
                if self.data is None:
                    if cols_to_filter is not None:
                        self.data = self.smooth_angular_accels(pd.read_csv(os.path.join(self.data_path, f)), cols_to_filter)
                    else:
                        self.data = pd.read_csv(os.path.join(self.data_path, f))
                else:
                    if cols_to_filter is not None:
                        self.data = pd.concat([self.data, self.smooth_angular_accels(pd.read_csv(os.path.join(self.data_path, f)), cols_to_filter)])
                    else:
                        self.data = pd.concat([self.data, pd.read_csv(os.path.join(self.data_path, f))])


        self.motor_speed_columns = self.get_column_names()[-9:-5]
        self.motor_derivative_columns = self.get_column_names()[-5:-1]

        self.length = self.data[self.state_columns].values.shape[0]

        self.calculate_state_dot_values()

    def get_time_values(self):
        return self.data['t'].values

    def get_state_data(self):
        rpms_squared = self.data[self.state_columns[-4:]].values**2
        # scale down rpms so network doesn't oversaturate and fill with nans
        rpms_squared *= np.max(self.data[self.state_columns[-4:]].values) / np.max(rpms_squared)
        vals = np.hstack([self.data[self.state_columns].values, rpms_squared])
        return vals

    def get_control_inputs(self):
        return self.data[self.motor_speed_columns].values

    def get_des_rpm_values(self):
        rpm_dot_vals = (np.diff(self.data[self.motor_speed_columns].values, axis=0)) / dt
        # copy last time step so rpm_dot_vals is same length as data
        rpm_dot_vals = np.vstack([rpm_dot_vals, rpm_dot_vals[-1,:]])

        return rpm_dot_vals / self.motor_time_constant + self.data[self.motor_speed_columns].values

    def get_battery_voltage_data(self):
        return self.data['vbat'].values

    def calculate_state_dot_values(self):
        # actual_ang_vels = (self.data[['ang vel x', 'ang vel y', 'ang vel z']].values[1:] - self.data[['ang vel x', 'ang vel y', 'ang vel z']].values[:-1]) / self.dt
        # actual_ang_vels = np.vstack([np.array([0.0, 0.0, 0.0]), actual_ang_vels])
        # self.state_dot_values = np.hstack([ self.data[['acc x', 'acc y', 'acc z']].values, actual_ang_vels])
        self.state_dot_values = self.data[['acc x', 'acc y', 'acc z', 'ang acc x', 'ang acc y', 'ang acc z']].values

    def saveData(self, filePath):
        np.savez(filePath, input=self.get_state_data(), labels=self.state_dot_values, control_inputs=self.get_control_inputs())

    # smoothed by applying moving average filter
    def smooth_angular_accels(self, data, cols_to_filter, plot_filtering=False):
        for col in cols_to_filter:

            new_data = np.convolve(data[col].values, np.ones(self.N)/self.N, mode='valid')

            # account for convolution output being smaller than data length
            #adding estimates for first few datapoints
            for i in range(self.N//2-1):
                old_datum = np.sum(data[col].values[:i+1]) / (i+1)
                new_data = np.insert(new_data, i, old_datum)

            #adding estimates for last few datapoints
            for i in range(self.N//2):
                old_datum = np.sum(data[col].values[-i-1:]) / (i+1)
                new_data = np.insert(new_data, -1, old_datum)

            if plot_filtering:
                t_vals = data['t'].values
                import matplotlib.pyplot as plt
                plt.plot(t_vals, data[col].values, label='Pre-filtering')
                plt.plot(t_vals, new_data, label='Post-filtering')
                plt.legend()
                plt.xlabel("Time [sec]")
                plt.ylabel("Angular Acceleration [rad/sec^2]")
                plt.show()

            pd_update = pd.DataFrame({col: new_data})
            data.update(pd_update)

        return data

    def set_data_filter_width(self, width):
        self.N = width

    def poly_fit_angular_accelerations(self):
        import matplotlib.pyplot as plt

        pts = 500

        poly = np.polyfit(self.get_time_values()[:pts], self.data['ang acc x'].values[:pts], 100)
        fit_func = np.poly1d(poly)
        plt.plot(self.get_time_values()[:pts], self.data['ang acc x'].values[:pts], label="actual")
        plt.plot(self.get_time_values()[:pts], fit_func(self.get_time_values()[:pts]), label="reg")
        plt.legend()
        plt.show()

class DynamicsDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y):

        # Assign data and label to self
        self.X = X
        self.Y = Y

        self.length = X.shape[0]

    def __len__(self):

        # Return length
        return self.length

    def __getitem__(self, index):

        ### Return data at index pair with context and label at index pair (1 line)
        return self.X[index, :], self.Y[index, :]

    @staticmethod
    def collate_fn(batch):

        # Select all data from batch
        batch_x = [x for x, y in batch]

        # Select all labels from batch
        batch_y = [y for x, y in batch]

        # Convert batched data and labels to tensors
        batch_x = torch.as_tensor(batch_x)
        batch_y = torch.as_tensor(batch_y)

        # Return batched data and labels
        return batch_x, batch_y
