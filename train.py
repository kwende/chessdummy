import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import IterableDataset
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import datetime

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # NOTE: taking it down to 2 convd layers made it bad. But, removing the # of channels seemed to help per layer
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, padding=2)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=4)
        self.cnn3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=4)
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 10)
        self.fc2 = nn.Linear(10, 64)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.cnn3(x)
        x = torch.relu(x)
        x = self.cnn4(x)
        x = torch.relu(x)
        x = self.global_pooling(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

model = CNN()

#cp = torch.load('D:/repos/chessdummy/best.ckpt')

#model.load_state_dict(cp['model_state_dict'])
model.cuda()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#optimizer.load_state_dict(cp['optimizer_state_dict'])

chess_dataset = IterableDataset.ChessDataset("e:/chess.db")

train_loader = torch.utils.data.DataLoader(dataset=chess_dataset, batch_size=500)
test_loader = torch.utils.data.DataLoader(dataset=chess_dataset, batch_size=5000)

# Train the model

n_epochs=1000
cost_list=[]
#accuracy_list=[]
N_test=5000
COST=0

def train_model(n_epochs):
    for epoch in range(n_epochs):
        #COST=0
        best_cost = -1.0
        print("Running...")
        for x, y in train_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            start = datetime.datetime.now()
            optimizer.step()
            end = datetime.datetime.now()
            cost = loss.data
            if best_cost == -1 or cost < best_cost:
                best_cost = cost
                print("saving...")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "Best.ckpt")
                print("...saved model")
        
        #cost_list.append(COST)
        # correct=0
        # #perform a prediction on the validation  data  
        # for x_test, y_test in test_loader:
        #     z = model(x_test)
        #     _, yhat = torch.max(z.data, 1)
        #     correct += (yhat == y_test).sum().item()
        # accuracy = correct / N_test
        #accuracy_list.append(accuracy)
     
train_model(n_epochs)
writer.flush()