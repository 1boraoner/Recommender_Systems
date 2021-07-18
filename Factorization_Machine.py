import torch
from torch.nn import Module, Embedding, Linear, MSELoss, CrossEntropyLoss
import pandas as pd
class Factorization_Machine(Module):

    def __init__(self, field_dims, fact_num):
        super(Factorization_Machine, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = Embedding(num_inputs, fact_num)
        self.fc = Embedding(num_inputs, 1)
        self.linear_layer = Linear(1, bias=True)

    def forward(self, x):
        square_of_sum = torch.sum(self.embedding(x), axis=1)**2
        sum_of_square = torch.sum(self.embedding(x)**2, axis=1)
        x = self.linear_layer(self.fc.sum(1)) + 0.5*(square_of_sum - sum_of_square).sum(1, keepdim=True)
        x = torch.nn.functional.sigmoid(x)
        return x



class AdvertisingData(torch.utils.data.Dataset):

    def __init__(self,path):

        df = pd.read_csv(path)
        


    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]
# load the Data

