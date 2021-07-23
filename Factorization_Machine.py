import torch
from torch.nn import Module, Embedding, Linear, BCEWithLogitsLoss, BCELoss, Parameter
import pandas as pd
import load_split_data as lsd


class Factorization_Machine(Module):

    def __init__(self, field_dims, fact_num, device):
        super(Factorization_Machine, self).__init__()
        self.device = device
        self.field_dims = field_dims
        num_inputs = int(sum(field_dims))

        self.embedding = Embedding(num_inputs, fact_num)
        self.fc = Embedding(num_inputs, 1)
        # self.linear_layer = Linear(in_features=1, out_features=1, bias=True)
        self.linear_layer = Linear(in_features=num_inputs, out_features=1, bias=True)

        self.V = Parameter(torch.randn(num_inputs, fact_num), requires_grad=True)

    def forward(self, x):
        gen_out = self.embedding(x.long())
        square_of_sum = (gen_out.sum(axis=1)) ** 2
        sum_of_square = (gen_out ** 2).sum(axis=1)
        inter = 0.5 * (square_of_sum.div(square_of_sum.max()) - sum_of_square.div(sum_of_square.max())).sum(axis=1,
                                                                                                            keepdim=True)
        lin = self.linear_layer(x.float())
        out = lin + inter
        out = torch.sigmoid(out)

        return out.squeeze(dim=1)

    def init_weights(self, m):
        if m == Embedding or m == Linear:
            torch.nn.init.normal(m.weight, mean=0.0, std=1.0)


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def evaluator(model, test_loader):
    loss_f = BCELoss()
    loss_record = []
    True_pred = 0

    for idx, data in enumerate(test_loader):
        model_preds = model(data[:, :-1])
        loss = loss_f(model_preds.float(), data[:, -1].float())
        True_pred += sum(model_preds.round().to(torch.int16) == data[:, -1].to(torch.int16))

        loss_record.append(loss)

    test_acc = (True_pred) / ((idx + 1) * (test_loader.batch_size))
    mean_loss = torch.tensor(loss_record).mean().to(torch.float64)
    return mean_loss, test_acc


def train_model(model, num_epochs, train_dl, test_dl, optimizer, loss_function):
    model_loss_history = []

    for i in range(num_epochs):
        epoch_history = []
        epoch_loss = 0

        for idx, data in enumerate(train_dl):
            optimizer.zero_grad()

            model_preds = model(data[:, :-1])
            # print(model_preds)

            batch_loss = loss_function(model_preds.float(), data[:, -1].float())
            # break
            epoch_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (idx + 1)
        epoch_history.append(epoch_loss)
        test_rmse, test_acc = evaluator(model, test_dl)

        model_loss_history.append((epoch_loss, test_rmse, test_acc))

        print(f" {i}th epoch loss: {epoch_loss}, test_set loss: {test_rmse}, test_acc = {test_acc} ")

    return model_loss_history


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
a, b, train_dl, test_dl = lsd.split_and_load_data(batch_size=1024,encoded=True, device=device)


FM = Factorization_Machine([943, 1682], fact_num=30, device=device)
FM.to(device)
FM.apply(FM.init_weights)
optimizer = torch.optim.Adam(lr=0.02, params=FM.parameters(), weight_decay=1e-5)

num_epochs = 20

loss_function = BCELoss()
hist = train_model(FM, num_epochs, train_dl, test_dl, optimizer, loss_function)

import matplotlib.pyplot as plt
import numpy as np
plt.plot(list(range(0,num_epochs)) ,hist)