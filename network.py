import torch
import torch.nn as nn
from torchsummary import summary

class SOFTNet(nn.Module):
    def __init__(self, in_channels=1):
        super(SOFTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=3, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels=5, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels=8, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.flatten = nn.Flatten()
        
    def forward(self, x1, x2, x3):
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool3(x1)
        x2 = self.conv2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool3(x2)
        x3 = self.conv3(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool3(x3)
        x = torch.cat((x1, x2, x3),1)
        x = self.maxpool2(x)
        x = self.flatten(x)
        return x
    
class MTSN(nn.Module):
    def __init__(self):
        super(MTSN, self).__init__()
        self.SOFTNet = SOFTNet()
        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = self.SOFTNet(x1, x2, x3)
        x2 = self.SOFTNet(x4, x5, x6)
        x = torch.cat((x1, x2),1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
# model = MTSN().cuda()
# summary(model, [(1,28,28),(1,28,28),(1,28,28),(1,28,28),(1,28,28),(1,28,28)])