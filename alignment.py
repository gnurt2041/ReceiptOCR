
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as torchvision_T

transform = torchvision_T.Compose([
    torchvision_T.Resize((400, 400)),
    torchvision_T.ToTensor(),
    torchvision_T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class AngleClassificationCNN(nn.Module):
    def __init__(self):
        super(AngleClassificationCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(512 * 12 * 12, 512)  # Update based on actual size
        self.fc2 = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(x1)
        x3 = F.relu(self.conv2(x2))
        x4 = self.pool(x3)
        x5 = F.relu(self.conv3(x4))
        x6 = self.pool(x5)
        x7 = F.relu(self.conv4(x6))
        x8 = self.pool(x7)
        x9 = F.relu(self.conv5(x8))
        x10 = self.pool(x9)

        # Flatten the tensor for fully connected layers
        x11 = x10.view(x10.size(0), -1)
        x12 = F.relu(self.fc1(x11))
        x13 = self.dropout(x12)
        x14 = self.fc2(x13)

        return x14, [x1, x3, x5, x7, x9]