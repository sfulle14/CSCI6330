import polars
import torch
import pytorch_lightning as pl
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class NeuralNet(pl.LightningModule):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(29, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch):
        X, y = batch
        y_hat = self(X)
        y_pred = (y_hat > 0.5).float()
        accuracy = (y_pred.squeeze() == y).float().mean()
        self.log("test_accuracy", accuracy)
        return accuracy


def train_neural_network(X_train, X_test, y_train, y_test):
    torch.set_float32_matmul_precision('medium')
    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=8)

    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)

    model = NeuralNet()

    trainer = pl.Trainer(max_epochs=5)

    trainer.fit(model, train_loader)

    trainer.test(model, test_loader)

    return 0 # todo
