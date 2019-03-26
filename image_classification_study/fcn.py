import torch.nn as nn
import fudge_grad

class FullyConnected(nn.Module):

    def __init__(self, num_classes=200, batch_size=256):
        super(FullyConnected, self).__init__()
        self.batch_size = batch_size
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Linear(224*224*3, int(224*224/64))
        self.layer2 = nn.Linear(int(224*224/64), num_classes)
        self.layer3 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.layer1(x.view(self.batch_size, -1))
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

class FudgellyConnected(nn.Module):

    def __init__(self, num_classes=200, batch_size=256):
        super(FudgellyConnected, self).__init__()
        self.batch_size = batch_size
        self.fudge = fudge_grad.FudgeGrad().apply
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Linear(224*224*3, int(224*224/64))
        self.layer2 = nn.Linear(int(224*224/64), num_classes)
        self.layer3 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.layer1(x.view(self.batch_size, -1))
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.fudge(x)
        x = self.layer3(x)
        return x

class FullyConnectedSingleDrop(nn.Module):

    def __init__(self, num_classes=200, batch_size=256, drop_p=0.5):
        super(FullyConnectedSingleDrop, self).__init__()
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Linear(224*224*3, int(224*224/64))
        self.layer2 = nn.Linear(int(224*224/64), num_classes)
        self.layer3 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.layer1(x.view(self.batch_size, -1))
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class FullyConnectedDoubleDrop(nn.Module):

    def __init__(self, num_classes=200, batch_size=256, drop_p=0.5):
        super(FullyConnectedDoubleDrop, self).__init__()
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Linear(224*224*3, int(224*224/64))
        self.layer2 = nn.Linear(int(224*224/64), num_classes)
        self.layer3 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.layer1(x.view(self.batch_size, -1))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x