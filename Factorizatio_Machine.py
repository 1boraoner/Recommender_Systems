import torch
from torch.nn import Module, Embedding, Linear, MSELoss, CrossEntropyLoss

class Factorization_Machine(Module):

    def __init__(self, field_dims, fact_num):
        super(Factorization_Machine, self).__init__()

