import torch
from torch.nn import Module, Embedding, Linear, MSELoss, CrossEntropyLoss,BCEWithLogitsLoss
import pandas as pd
import load_split_data as lsd


class Factorization_Machine(Module):

    def __init__(self, field_dims, fact_num, device):

        super(Factorization_Machine, self).__init__()
        self.device = device
        self.field_dims = field_dims
        num_inputs = int(sum(field_dims))

        self.embedding = Embedding(num_inputs, fact_num)
        #self.fc = Embedding(num_inputs, 1)
        self.linear_layer = Linear(in_features=num_inputs, out_features=1, bias=True)

    def forward(self, x):

        gen_out = self.embedding(x.long())

        square_of_sum = (gen_out.sum(1))**2
        sum_of_square = (gen_out**2).sum(1)

        inter = 0.5 * (square_of_sum - sum_of_square).sum(1, keepdim=True)
        lin = self.linear_layer(x)
        out = lin + inter

        out = sigmoid(out)
        return out.squeeze(dim=1)

    def init_weights(self,m):
      if m == Embedding or m == Linear:
        torch.nn.init.normal_xavier_(m.weight)


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def evaluator(model, test_loader):
    mse = BCEWithLogitsLoss()
    loss_record = []
    for data in test_loader:
        model_preds = model(data[:, :-1])
        batch_loss = loss_function(model_preds.float(), data[:, -1].float())

        loss = torch.sqrt(mse(model_preds, data[:, -1]))
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

            optimizer.zero_grad()

            model_preds = model(data[:, :-1])
            batch_loss = loss_function(model_preds.float(), data[:,-1].float())

            epoch_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (idx + 1)
        epoch_history.append(epoch_loss)
        test_rmse = evaluator(model, test_dl)

        model_loss_history.append((epoch_loss, test_rmse))

        print(f" {i}th epoch loss: {epoch_loss}, test_set loss: {test_rmse}")

    return model_loss_history

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


a, b, train_dl, test_dl = lsd.split_and_load_data(batch_size=2048,encoded=True, device=device)
FM = Factorization_Machine([943, 1682], fact_num=20, device=device)
FM.to(device)
FM.apply(FM.init_weights)
optimizer = torch.optim.Adam(lr=0.02, params=FM.parameters(), weight_decay=1e-5)

num_epochs = 1

loss_function = BCEWithLogitsLoss()
hist = train_model(FM, num_epochs, train_dl, test_dl, optimizer, loss_function)




# def OneHotEncoder(original, num_feats):
#     encoded_users = torch.zeros(size=(original.shape[0], num_feats[0]))
#     encoded_movies = torch.zeros(size=(original.shape[0], num_feats[1]))
#
#     recommended = torch.zeros(size=(original.shape[0], 1))
#
#     for row in range(original.shape[0]):
#         encoded_users[row, original[row, 0].long()] = 1
#         encoded_movies[row, original[row, 1].long()] = 1
#         recommended[row, 0] = 1 if original[row, 2] >= 3 else 0
#
#     encoded_tensor = torch.cat((encoded_users, encoded_movies), dim=1)
#     return encoded_tensor, recommended


# def L2Loss(r_hat, r_true):
    # return 0.5 * torch.sum((r_hat - r_true) ** 2)
