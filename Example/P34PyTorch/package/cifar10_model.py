from torch import nn
import torch.nn.functional as F
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 ,10 ,3 ,1 ,1),  # 32x32x3 -> 30x30x10
            nn.ReLU(),                  # 30x30x10 -> 30x30x10
            nn.MaxPool2d(2 ,2),         # 30x30x10 -> 15x15x10
            nn.Conv2d(10 ,20 ,3 ,1 ,1), # 15x15x10 -> 13x13x20
            nn.ReLU(),                  # 13x13x20 -> 13x13x20
            nn.MaxPool2d(2 ,2),         # 13x13x20 -> 6x6x20
            nn.Conv2d(20 ,40 ,3 ,1 ,1), # 6x6x20 -> 4x4x40
            nn.ReLU(),                  # 4x4x40 -> 4x4x40
            nn.MaxPool2d(2, 2),         # 4x4x40 -> 2x2x40
            Reshape(-1 ,40 *4 *4),      # 2x2x40 -> 160
            nn.Dropout(0.2),            # 160 -> 160
            nn.Linear(40*4*4,100) ,     # 160 -> 100
            nn.ReLU(),                  # 100 -> 100
            nn.Dropout(0.2),            # 100 -> 100
            nn.Linear(100, 10),         # 100 -> 10
            nn.LogSoftmax(dim=1)        # 10 -> 10
        )
        # Reshape：將輸入的張量變形為指定的形狀
        # Reshape (-1, 40*4*4)：
        # Reshape -1表示自動計算，40*4*4表示輸出的張量形狀

    def forward(self, x):
        op = self.model(x)
        return op