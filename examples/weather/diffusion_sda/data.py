import torch
import zarr
import numpy as np
from torch.utils.data import Dataset

from physicsnemo.utils.zenith_angle import cos_zenith_angle


class HRRRSurfaceDataset(Dataset):

    def __init__(self, zarr_root: zarr.group, time_indices: np.array):

        self.root = zarr_root
        self.idx = time_indices

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, idx):

        time_idx = self.idx[idx]
        time_stamp = self.root['time'][time_idx]

        
        


