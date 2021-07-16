import torch
from torch.nn import Embedding, MSELoss
from load_split_data import split_and_load_data
import matplotlib.pyplot as plt
import numpy as np

# model implementation

class MatrixFactorization(torch.nn.Module):

    def __init__(self, latent_factor, user_num, item_num):
        super().__init__()
        self.Pmatrix = Embedding(user_num, latent_factor)
        self.Qmatrix = Embedding(item_num, latent_factor)
        self.user_bias = Embedding(user_num, 1)
        self.item_bias = Embedding(item_num, 1)

    def forward(self, users, items):
        Pm = self.Pmatrix(users)
        Qm = self.Qmatrix(items)
        bu = self.user_bias(users)
        bi = self.item_bias(items)
        Rm = (Pm * Qm).sum(axis=1) + bu.squeeze() + bi.squeeze()
        return Rm.flatten()

    def init_weights(self,m):
        if m == Embedding:
            torch.nn.init.normal_(m.weight, mean=0, std=0.01)

def L2Loss(r_hat, r_true):
    return 0.5 * torch.sum((r_hat - r_true) ** 2)


# evaluation measures

def evaluator(model, test_loader):
    mse = MSELoss()
    loss_record = []
    for data in test_loader:
        batch_users = data[:, 0].long()
        batch_movies = data[:, 1].long()
        r_true = data[:, 2]

        r_preds = model(batch_users, batch_movies)
        loss = torch.sqrt(mse(r_preds, r_true))
        loss_record.append(loss)

    mean_loss = torch.tensor(loss_record).mean().to(torch.float64)
    return mean_loss


def train_model(model, train_dl, test_dl, loss_function, optimizer, num_epochs):
    model_loss_history = []
    for epoch in range(num_epochs):

        epoch_history = []
        epoch_loss = 0

        for idx, data in enumerate(train_dl):

            model_preds = model(data[:, 0].long(), data[:, 1].long())
            batch_loss = sum(0.5*(model_preds - data[:, 2].float()) ** 2)/512

            epoch_loss += batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (idx + 1)
        epoch_history.append(epoch_loss)
        test_rmse = evaluator(model, test_dl)

        model_loss_history.append((epoch_loss,test_rmse))

        print(f" {epoch}th epoch loss: {epoch_loss}, test_set loss: {test_rmse}")

    return model_loss_history


device = "gpu" if torch.cuda.is_available() else "cpu"
num_users, num_items, train_dl, test_dl = split_and_load_data(test_ratio=0.1, batch_size=512)
model = MatrixFactorization(30, num_users, num_items)
model.apply(model.init_weights)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.02, weight_decay=1e-5)
loss_function = L2Loss

hist = train_model(model, train_dl, test_dl, loss_function, optimizer, 20)



print(model(torch.tensor([20]),torch.tensor([30])))
print(model(torch.tensor([0]),torch.tensor([167])))
print(model(torch.tensor([0]),torch.tensor([4])))


plt.plot(np.arange(0,20), hist)
plt.show()