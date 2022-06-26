import torch
from torch.utils.data import Dataset

# Convert dataset into tensor for model training or testing
class OFFSpottingDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]
    
    # Convert into two 3-stream data
    def __getitem__(self, index):
        x1 = torch.reshape(self.tensors[0][index][0].to('cuda'), (1,28,28))
        x2 = torch.reshape(self.tensors[0][index][1].to('cuda'), (1,28,28))
        x3 = torch.reshape(self.tensors[0][index][2].to('cuda'), (1,28,28))
        x4 = torch.reshape(self.tensors[1][index][0].to('cuda'), (1,28,28))
        x5 = torch.reshape(self.tensors[1][index][1].to('cuda'), (1,28,28))
        x6 = torch.reshape(self.tensors[1][index][2].to('cuda'), (1,28,28))
        return x1, x2, x3, x4, x5, x6
