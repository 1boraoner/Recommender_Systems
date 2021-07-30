import torch
from torch.nn import Module, Embedding, Linear, BCELoss
import load_split_data as lsd


class DeepFM(Module):

    def __init__(self, field_dims, fact_num, mlp_dims, drop_out=0.1, device):
        super(DeepFM, self).__init__()
        self.device = device
        self.
