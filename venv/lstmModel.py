import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LSTM(nn.Module):
    def __init__(self, d_in=6, h=64, d_out=1, N=8, device=torch.device("cpu")):
        super().__init__()
        self.h = h
        self.lstm = nn.LSTM(d_in, h).to(device)
        self.out = nn.Linear(h, d_out).to(device)
        self.hidden = (torch.randn(1, N, self.h).to(device), torch.randn(1, N, self.h).to(device))
        self.device = device

    def forward(self, x, batch_size):
        self.hidden = (torch.randn(1, batch_size, h).to(self.device), torch.randn(1, batch_size, h).to(self.device))
        output, self.hidden = self.lstm(x, self.hidden)
        predictions = self.out(output)
        return predictions[-1]


'''gauge_id = 13331500
training_data = Streamflow(gauge_id, "training")
validation_data = Streamflow(gauge_id, "validation")

training_loader = training_data.data_loader(batch_size=8)'''

rand_coin = 'ETH'
self.data_min = pd.read_csv(f'data/Binance_{rand_coin}USDT_minute.csv', sep=',',
                            usecols=['date', 'open', 'high', 'low', 'close', f'Volume {rand_coin}',
                                     'Volume USDT', 'tradecount'], skiprows=1)
batch_size, dIn, h, dOut = 8, 6, 64, 1
loader = DataLoader(self.training_date, batch_size=batch_size, shuffle=True)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = LSTM(dIn, h, dOut, batch_size, dev)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 40
n = len(training_loader)

for epoch in range(num_epochs):
    training_loss = 0

    for (batchX, batchY) in training_loader:
        optimizer.zero_grad()
        yPred = model(batchX, batch_size)
        loss = criterion(yPred, batchY[-1])
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    print('Training loss:', epoch, training_loss / n)
    if epoch % 10 == 9:
        x, y = validation_data.get_range()
        val_yPred = model(x, 1)
        val_loss = criterion(val_yPred, y[-1])
        print('Validation loss:', epoch, val_loss.item())

torch.save(model.state_dict(), "lstmPricePredict")
