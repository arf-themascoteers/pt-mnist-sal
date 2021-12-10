import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.linear1 = nn.Linear(784, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)