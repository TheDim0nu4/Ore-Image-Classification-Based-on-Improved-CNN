from torch import nn



class SENetModule(nn.Module):

    def __init__(self, in_channels):

        super(SENetModule, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):

        if isinstance(x, list): 
            x = x[-1]

        batch, channels, _, _ = x.shape
        se = self.global_avg_pool(x).view(batch, channels)
        se = self.fc(se).view(batch, channels, 1, 1)

        return x * se


