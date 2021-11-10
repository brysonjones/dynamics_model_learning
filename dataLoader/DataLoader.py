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
        self.H = np.vstack([np.zeros(3), np.eye(3)]) # constant matrix for quaternion math

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

        print("Calculating state derivatives... (note: this may take a while, estimate: {0:0.2f} [sec])".format(self.length / 800))
        self.calculate_state_dot_values()

    def get_column_names(self):
        return self.data.keys().values

    def load_selected_data(self, selected_data):
        self.selected_data = selected_data

        self.data = None
        flights = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]

        if len(flights) == 0:
            print("ERROR! No csv files found with the names provided: \n", selected_data)

        for f in flights:
            if f[7:-10] in selected_data:
                if self.data is None:
                    self.data = pd.read_csv(os.path.join(self.data_path, f))
                else:
                    self.data = pd.concat([self.data, pd.read_csv(os.path.join(self.data_path, f))])

        self.motor_speed_columns = self.get_column_names()[-9:-5]
        self.motor_derivative_columns = self.get_column_names()[-5:-1]

        self.length = self.data[self.state_columns].values.shape[0]

        print("Calculating state derivatives... (note: this may take a while, estimate: {0:0.2f} [sec])".format(self.length / 800))
        self.calculate_state_dot_values()

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

    def get_battery_voltage_data(self):
        return self.data['vbat'].values

    def calculate_state_dot_values(self):
        # state_dot_columns = ['vel x', 'vel y', 'vel z', 'acc x', 'acc y', 'acc z', d_quat-x, d_quat-y, d_quat-z,
                             # d_quat-w, 'ang acc x', 'ang acc y', 'ang acc z', 'dmot 1' 'dmot 2' 'dmot 3' 'dmot 4']
        self.state_dot_values = self.data[['vel x', 'vel y', 'vel z', 'acc x', 'acc y', 'acc z']].values
        quat_deriv_vals = []

        start = time.time()
        for i in range(self.length):
            # quat_deriv_vals.append(0.5 * self.quat_L(self.data[['quat x', 'quat y', 'quat z', 'quat w']].values[i,:]) @ self.H @ self.data[['ang vel x', 'ang vel y', 'ang vel z']].values[i,:])
            quat_deriv_vals.append( self.quat_dot(self.data[['quat x', 'quat y', 'quat z', 'quat w']].values[i,:], self.data[['ang vel x', 'ang vel y', 'ang vel z']].values[i,:]) )
        self.state_dot_values = np.hstack([ self.state_dot_values, np.stack(quat_deriv_vals) ])
        print("Time to calculate quaternion derivatives: {0:0.2f} [sec]".format(time.time() - start))
        self.state_dot_values = np.hstack([self.state_dot_values, self.data[['ang acc x', 'ang acc y', 'ang acc z', 'dmot 1', 'dmot 2', 'dmot 3', 'dmot 4']].values])

    # L operator for quaternions that Zac covered in lec 7, used to calculate quaternion derivatives
    @staticmethod
    def quat_dot(q, w):
        q_vec_hat = np.array([[0, -q[3], q[2]], [q[3], 0, -q[1]], [-q[2], q[1], 0]])

        L = np.zeros((4,4))
        L[0,0] = q[0]
        L[0,1:] = -q[1:]
        L[1:,0] = q[1:]
        L[1:,1:] = q[0]*np.eye(3) + q_vec_hat

        return 0.5 * L @ np.vstack([np.zeros(3), np.eye(3)]) @ w

    # hat operator for a length 3 vector
    @staticmethod
    def hat(x):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    def saveData(self, filePath):
        np.savez(filePath, input=self.get_state_data(), output=self.state_dot_values, control_inputs=self.get_control_inputs())

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
