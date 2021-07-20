import torch
from torch.nn import Module, Embedding, Linear, MSELoss, CrossEntropyLoss
import pandas as pd
import load_split_data as lsd


class Factorization_Machine(Module):

    def __init__(self, field_dims, fact_num):
        super(Factorization_Machine, self).__init__()
        self.field_dims = field_dims
        num_inputs = int(sum(field_dims))
        self.embedding = Embedding(num_inputs, fact_num)
        self.fc = Embedding(num_inputs, 1)
        self.linear_layer = Linear(in_features=num_inputs, out_features=1, bias=True)

    def forward(self, x):
        encoded_tensor, recommended = OneHotEncoder(x.long(), self.field_dims)

        square_of_sum = torch.pow(self.embedding(encoded_tensor.long()).sum(axis=1), 2)
        sum_of_square = torch.pow(self.embedding(encoded_tensor.long()), 2).sum(axis=1)
        interaction_dif = 0.5 * (square_of_sum - sum_of_square).sum(axis=1)
        l = self.linear_layer(encoded_tensor)
        out = l + interaction_dif.unsqueeze(dim=1)
        out = torch.sigmoid(out)

        return out, recommended


def OneHotEncoder(original, num_feats):
    encoded_users = torch.zeros(size=(original.shape[0], num_feats[0]))
    encoded_movies = torch.zeros(size=(original.shape[0], num_feats[1]))

    recommended = torch.zeros(size=(original.shape[0], 1))

    for row in range(original.shape[0]):
        encoded_users[row, original[row, 0].long()] = 1
        encoded_movies[row, original[row, 1].long()] = 1
        recommended[row, 0] = 1 if original[row, 2] >= 3 else 0

    encoded_tensor = torch.cat((encoded_users, encoded_movies), dim=1)
    return encoded_tensor, recommended


def L2Loss(r_hat, r_true):
    return 0.5 * torch.sum((r_hat - r_true) ** 2)


def evaluator(model, test_loader):
    mse = MSELoss()
    loss_record = []
    for data in test_loader:
        model_preds, true_recommend = model(data.long())
        loss = torch.sqrt(mse(model_preds, true_recommend))
        loss_record.append(loss)

    mean_loss = torch.tensor(loss_record).mean().to(torch.float64)
    return mean_loss


def train_model(model, num_epochs, train_dl, test_dl, optimizer, loss_function):
    model_loss_history = []

    for i in range(num_epochs):
        epoch_history = []
        epoch_loss = 0
        i = 0
        for idx, data in enumerate(train_dl):
            print(i)
            i += 1

            model_preds, true_recommend = model(data.long())
            batch_loss = sum(0.5 * (model_preds - true_recommend) ** 2) / 2048

            epoch_loss += batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (idx + 1)
        epoch_history.append(epoch_loss)
        test_rmse = evaluator(model, test_dl)

        model_loss_history.append((epoch_loss, test_rmse))

        print(f" {i}th epoch loss: {epoch_loss}, test_set loss: {test_rmse}")

    return model_loss_history


# a, b, train_dl, test_dl = lsd.split_and_load_data(batch_size=2048)
# FM = Factorization_Machine([943, 1682], fact_num=20)
# optimizer = torch.optim.Adam(lr=0.02, params=FM.parameters(), weight_decay=1e-5)
#
# num_epochs = 10
#
# loss_function = L2Loss
# hist = train_model(FM, num_epochs, train_dl, test_dl, optimizer, loss_function)

x = torch.rand(3,1, dtype=torch.long)
y = torch.rand(3,1, dtype=torch.long)

f = CrossEntropyLoss()

print(f(x,y))
