import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):

    def __init__(self, user_ids, item_ids, scores, device, ohencoded=False, ctr=False):
        self.datas = torch.zeros(size=(len(user_ids), 3), device=device)
        self.datas[:, 0] = torch.tensor(user_ids)
        self.datas[:, 1] = torch.tensor(item_ids)
        self.datas[:, 2] = torch.tensor(scores)
        if ctr == True:
          for i in range(self.datas.shape[0]):
            self.datas[i, 2] = 1 if self.datas[i, 2] > 3 else 0
        self.length = self.datas.shape[0]

        if ohencoded:
            encoded_users = torch.zeros(size=(self.datas.shape[0], 943), device = device)
            encoded_movies = torch.zeros(size=(self.datas.shape[0], 1682),device = device)
            recommended = torch.zeros(size=(self.datas.shape[0], 1),device = device)

            for row in range(self.datas.shape[0]):
                encoded_users[row, self.datas[row, 0].long()] = 1
                encoded_movies[row, self.datas[row, 1].long()] = 1
                recommended[row, 0] = 1 if self.datas[row, 2] >= 3 else 0

            self.datas = torch.cat((encoded_users, encoded_movies,recommended), dim=1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        return self.datas[idx]turn self.datas[idx]