import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AlzheimerModel(nn.Module):

    def __init__(self, num_classes=4):
        super(AlzheimerModel, self).__init__()

        # Loading the pre-trained ResNet152 weights
        self.backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

        # Freezing the backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Calculating the number of input features incoming from conv layers to the FC layer.
        num_features = self.backbone.fc.in_features

        self.BatchNorm = nn.BatchNorm1d(num_features)
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.backbone(x)
        x = self.BatchNorm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.drop2(x)
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)
        return x