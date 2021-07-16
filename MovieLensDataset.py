import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):

    def __init__(self, user_ids, item_ids, scores):
        self.datas = torch.zeros(size=(len(user_ids), 3))
        self.datas[:, 0] = torch.tensor(user_ids)
        self.datas[:, 1] = torch.tensor(item_ids)
        self.datas[:, 2] = torch.tensor(scores)
        self.length = self.datas.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        return self.datas[idx]

