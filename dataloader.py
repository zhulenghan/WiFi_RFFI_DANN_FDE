from torch.utils.data import Dataset, DataLoader
import torch
import utils
import numpy as np


class DatasetBase(Dataset):
    def __init__(self, h5_group, transform=None):
        self.transform = transform
        self.data = h5_group['IQ_Samples']
        self.label = h5_group['Station']
        self.cfo = h5_group['CFO']
        self.csi = h5_group['CSI']

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        cfo = self.cfo[index]
        csi = self.csi[index]

        if self.transform == 'legacy_preamble':
            data = utils.legacy_preable(data)

        if self.transform == 'cfo_fde_40cbw':
            if len(data) >= 1280:
                data = utils.cfo_fde_40cbw(data, cfo, csi)
            else:
                raise Exception('Package format not correct')

        data = torch.from_numpy(data).float()
        label = torch.from_numpy(np.array(label)).long()

        return data, label

    def __len__(self):
        return len(self.label)


class DannDataset(DatasetBase):
    def __init__(self, h5_group, transform=None):
        super(DannDataset, self).__init__(h5_group=h5_group, transform=transform)
        self.domain = h5_group['Domain']

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        cfo = self.cfo[index]
        csi = self.csi[index]
        domain = self.domain[index]

        if self.transform == 'cfo_fde_40cbw':
            if len(data) >= 1280:
                data = utils.cfo_fde_40cbw(data, cfo, csi)
            else:
                raise Exception('Package format not correct')

        data = torch.from_numpy(data).float()
        label = torch.from_numpy(np.array(label)).long()
        domain = torch.from_numpy(np.array(domain)).long()

        return data, label, domain

    def __len__(self):
        return len(self.label)
