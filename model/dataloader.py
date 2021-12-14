
import torch
import torch.utils.data
import numpy as np


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

    def collate_fn(batch):

        # Select all data from batch
        batch_x = [x for x, y in batch]

        # Select all labels from batch
        batch_y = [y for x, y in batch]

        # Convert batched data and labels to tensors
        batch_x = torch.as_tensor(np.asarray(batch_x))
        batch_y = torch.as_tensor(np.asarray(batch_y))

        # Return batched data and labels
        return batch_x, batch_y
