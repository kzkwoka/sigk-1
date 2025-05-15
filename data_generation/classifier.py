import torch.nn as nn
from torchvision import models


class ClassifierModel(nn.Module):
    def __init__(self, num_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get ResNet50 model with ImageNet weights
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        num_ftrs = self.model.fc.in_features
        # Fully connected layer with num_labels classes

        self.model.fc = nn.Linear(num_ftrs, num_labels)
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = ClassifierModel(5)
    print(model)
