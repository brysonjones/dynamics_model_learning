import os
import pandas as pd

class DataLoader(object):

    def __init__(self, path_to_data):
        self.data_path = path_to_data
        self.data = None #stores pandas dataframes
        self.selected_data = None #just filenames to be loaded in (see self.load_selected_data())

        self.state_columns = ['ang acc x', 'ang acc y', 'ang acc z', 'ang vel x', 'ang vel y',
                            'ang vel z', 'quat x', 'quat y', 'quat z', 'quat w', 'acc x', 'acc y',
                            'acc z', 'vel x', 'vel y', 'vel z', 'pos x', 'pos y', 'pos z', 'mot 1',
                            'mot 2', 'mot 3', 'mot 4']

        self.motor_speed_columns = ['mot1', 'mot 2', 'mot 3', 'mot 4']
        self.motor_derivative_columns = ['dmot 1', 'dmot 2', 'dmot 3', 'dmot 4']


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
        self.data = None
        flights = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]

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
        # rpm_dot = k*(mot_des - mot_curr)
        # rpm_dot / k + mot_curr = mot_des


if __name__ == '__main__':
    DL = DataLoader("processed_data/")
    DL.load_easy_data()
    print(DL.get_column_names())
    test = DL.get_state_data()
    print(test.shape)
