import numpy as np
import torch
from torch.nn import Module, Sequential, Embedding, Linear, Dropout, ReLU, Tanh, BatchNorm1d
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
            self.mlp.add_module(name=f"norm{i}", module=BatchNorm1d(dim))
            self.mlp.add_module(f"relu{i}", ReLU())
            self.mlp.add_module(f"dropout{i}", Dropout(p=drop_rate))
            input_dim = dim

        self.mlp.add_module(f"linear_last", Linear(in_features=input_dim, out_features=1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        square_sum = torch.sum(embedded, dim=1) ** 2
        sum_square = torch.sum(embedded ** 2, dim=1)
        concat_embed = embedded.view(-1, self.embed_output_dim)
        x = self.linear_layer(self.fc(x).sum(axis=1)) + 0.5 * (square_sum - sum_square).sum(1, keepdim=True) + self.mlp(
            concat_embed)
        # nn_forward_feed = self.mlp(concat_embed)
        # x = x + nn_forward_feed

        x = self.sigmoid(x)
        return x.squeeze(dim=1)

    def init_weights(self, m):
        if isinstance(m, Embedding) or isinstance(m, Linear):
            torch.nn.init.xavier_normal_(m.weight)


def evaluator(model, test_loader, loss_function):
    loss_f = loss_function
    loss_record = []
    True_pred = 0

    for idx, data in enumerate(test_loader):
        model_preds = model(data[:, :-1].long())
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
        test_rmse, test_acc = evaluator(model, test_dl, loss_function)
        train_rmse, train_acc = evaluator(model, train_dl, loss_function)
        for idx, data in enumerate(train_dl):
            optimizer.zero_grad()

            model_preds = model(data[:, :-1].long())

            batch_loss = loss_function(model_preds.float(), data[:, -1].float())

            epoch_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (idx + 1)
        epoch_history.append(epoch_loss)

        model_loss_history.append((epoch_loss, test_rmse, test_acc, train_acc))

        print(
            f" {i}th epoch loss: {epoch_loss}, train_set acc: {train_acc}, test_set loss: {test_rmse}, test_acc = {test_acc} ")

    return model_loss_history

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

batch_size = 1024
un, mn, train_dl, test_dl = lsd.split_and_load_data(device=device, test_ratio=0.25, batch_size=batch_size, encoded=False, ctr=True)

model = DeepFM(field_dims=[un,mn], fact_num=10, mlp_dims=[200,100,50,25,10], device=device, drop_rate=0.1)
model.to(device)
model.apply(model.init_weights)
optimizer = torch.optim.Adam(lr=0.005, params=model.parameters(), weight_decay=1e-5)

loss_function = torch.nn.BCELoss()
hist = train_model(model, 15, train_dl, test_dl, optimizer, loss_function)


import matplotlib.pyplot as plt
plt.plot(hist)
plt.ylim([0, 1])
plt.show()