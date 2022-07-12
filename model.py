import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

class net(nn.Module):
    def __init__(self):
        super().__init__()
        inception_model = models.inception_v3(pretrained = True)
        inception_model.fc = nn.Linear(2048, 64)
        self.feature_extractor = inception_model

    def forward(self, x):
        x = transforms.functional.resize(x, size=[224, 224])
        x = x/255.0
        x = transforms.functional.normalize(x, mean = [0.5,0.5,0.5],
                                            std = [0.5,0.5,0.5])

        return self.feature_extractor(x).logits

if __name__ == '__main__':
    model = net()
    model.eval()
    saved_model = torch.jit.script(model)
    saved_model.save('weight/inception_v3.pt')