import numpy as np
import torch
from torch.nn import Module, Sequential, Embedding, Linear, Dropout, ReLU
import load_split_data as lsd


class DeepFM(Module):

    def __init__(self, field_dims, fact_num, mlp_dims, drop_rate=0.1, device=torch.device("cpu")):
        super(DeepFM, self).__init__()
        self.device = device
        self.field_dims = field_dims
        num_inputs = int(sum(field_dims))

        self.embedding = Embedding(num_inputs, fact_num)
        self.fc = Embedding(num_inputs, 1)
        self.linear_layer = Linear(in_features=1, out_features=1, bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * fact_num
        self.mlp = Sequential()
        for i, dim in enumerate(mlp_dims):
            self.mlp.add_module(name=f"linear{i}", module=Linear(in_features=input_dim, out_features=dim, bias=True))
            self.mlp.add_module(f"relu{i}", ReLU())
            self.mlp.add_module(f"dropout{i}", Dropout(p=drop_rate))
            input_dim = dim

        self.mlp.add_module(f"linear_last", Linear(in_features=input_dim, out_features=1))

    def forward(self, x):
        embedded = self.embedding(x)
        square_sum = torch.sum(embedded, axis=1) ** 2
        sum_square = torch.sum(embedded ** 2, axis=1)
        ins = embedded.view(-1, self.embed_output_dim)
        x = self.linear_layer(self.fc(x).sum(axis=1))

        x = x + 0.5 * (square_sum - sum_square).sum(1, keepdim=True)

        #x = x+ self.mlp(ins)
        x = torch.sigmoid(x)
        return x

    def init_weights(self, m):
        if m == Embedding or m == Linear:
            torch.nn.init.xavier_normal_(m.weight, gain=1.0)


batch_size = 2048
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

_, _, train_dl, test_dl = lsd.split_and_load_data(device=device, test_ratio=0.1, batch_size=batch_size, encoded=False)

model = DeepFM(field_dims=[943, 1682], fact_num=10, mlp_dims=[30, 20, 10], device=device)
model.to(device)
model.apply(model.init_weights)

optimizer = torch.optim.Adam(lr=0.01, params=model.parameters(), weight_decay=1e-5)
