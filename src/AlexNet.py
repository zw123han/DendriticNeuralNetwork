# https://medium.com/analytics-vidhya/alexnet-a-simple-implementation-using-pytorch-30c14e8b6db2
# Modifier a little bit to make it accept MNIST and return the probability of most-like-to-be digit.
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels= 32, kernel_size= 3, padding=1 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 2304, out_features= 1024)
        self.fc2  = nn.Linear(in_features= 1024, out_features= 512)
        self.fc3 = nn.Linear(in_features=512 , out_features=10)


    def forward(self,x):
        if len(x.shape) == 2:
          x = x.view(x.shape[0], 1, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(torch.argmax(x, dim=1).unsqueeze(1).shape)
        # print(torch.max(x, dim=1, keepdim = True))
        return torch.max(x, dim=1, keepdim = True) # Not sure if it's correct?