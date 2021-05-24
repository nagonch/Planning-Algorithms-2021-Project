import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, X_dim=6, c_dim=133, h_Q_dim=512, z_dim=3, h_P_dim=512):
        super(Network, self).__init__()

        self.dense_Q1 = nn.Linear(X_dim + c_dim, h_Q_dim)
        self.act_Q1 = nn.ReLU()
        self.dropout_Q1 = nn.Dropout()

        self.dense_Q2 = nn.Linear(h_Q_dim, h_Q_dim)
        self.act_Q2 = nn.ReLU()

        self.z_mu = nn.Linear(h_Q_dim, z_dim)
        self.z_logvar = nn.Linear(h_Q_dim, z_dim)

        self.dense_P1 = nn.Linear(z_dim + c_dim, h_P_dim)
        self.act_P1 = nn.ReLU()
        self.dropout_P1 = nn.Dropout()

        self.dense_P2 = nn.Linear(h_P_dim, h_P_dim)
        self.act_P2 = nn.ReLU()

        self.y = nn.Linear(h_P_dim, X_dim)

    def forward(self, X, c):
        input = torch.cat((X, c), axis=1)
        input = self.dense_Q1(input)
        input = self.act_Q1(input)
        input = self.dropout_Q1(input)

        input = self.dense_Q2(input)
        input = self.act_Q2(input)

        z_mu = self.z_mu(input)
        z_logvar = self.z_logvar(input)

        eps = torch.normal(torch.zeros_like(z_mu), torch.ones_like(z_mu))
        z = z_mu + torch.exp(z_logvar/2) * eps
        inputs_P = torch.cat((z, c), axis=1)

        inputs_P = self.dense_P1(inputs_P)
        inputs_P = self.act_P1(inputs_P)
        inputs_P = self.dropout_P1(inputs_P)

        inputs_P = self.dense_P2(inputs_P)
        inputs_P = self.act_P2(inputs_P)

        y = self.y(inputs_P)

        return y, z_mu, z_logvar

    def infer(self, Z, y):
        inputs_P = torch.cat((Z, y), axis=1)

        inputs_P = self.dense_P1(inputs_P)
        inputs_P = self.act_P1(inputs_P)
        inputs_P = self.dropout_P1(inputs_P)

        inputs_P = self.dense_P2(inputs_P)
        inputs_P = self.act_P2(inputs_P)

        y = self.y(inputs_P)

        return y


if __name__ == "__main__":
    pass
