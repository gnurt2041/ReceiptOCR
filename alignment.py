
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
        layers = [32, 64, 128, 256, 512]
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            ) for in_ch, out_ch in zip([3] + layers[:-1], layers)]
        )
        self.fc1 = nn.Linear(512 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        feature_maps = []  # List to store feature maps
        for conv in self.convs:
            x = conv(x)
            feature_maps.append(x)  # Save the output of each convolutional layer

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)

        return output, feature_maps  # Return both the final output and feature maps
