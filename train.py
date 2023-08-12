import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import IterableDataset

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding=2)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=2)
        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=2)
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.cnn3(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

model = CNN()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

chess_dataset = IterableDataset.ChessDataset("e:/chess.db")

train_loader = torch.utils.data.DataLoader(dataset=chess_dataset, batch_size=100)
test_loader = torch.utils.data.DataLoader(dataset=chess_dataset, batch_size=5000)

# Train the model

n_epochs=3
cost_list=[]
accuracy_list=[]
N_test=5000
COST=0

def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST=0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST+=loss.data
        
        cost_list.append(COST)
        correct=0
        #perform a prediction on the validation  data  
        for x_test, y_test in test_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
     
train_model(n_epochs)