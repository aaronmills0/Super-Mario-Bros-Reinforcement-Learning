import torch.nn as nn

class DuelingDQN(nn.Module):

    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8,8), stride=(4,4)), # 84 x 84 x 4 -> 20 x 20 x 32
            nn.ReLU(inplace=True), # Conv + ReLU
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2)), # 20 x 20 x 32 -> 8 x 8 x 64
            nn.ReLU(inplace=True), # Conv + ReLU
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1)), # 8 x 8 x 64 -> 7 x 7 x 64
            nn.ReLU(inplace=True), # Conv + ReLU
        )

        self.advantage = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
    
    def aggregate_layer(V, A):
        return V + (A - A.mean())
  
    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), - 1)
        A = self.advantage(X)
        V = self.value(X)
        X = self.aggregate_layer(V, A)
        return X
