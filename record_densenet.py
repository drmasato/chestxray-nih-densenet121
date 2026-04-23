"""DenseNet-121の既存結果をベンチマークに記録"""
import torch
import torch.nn as nn
from torchvision import models
from benchmark import record_result, get_test_loader

class ChestXrayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(nn.Linear(1024, 14))
    def forward(self, x):
        return self.model(x)

device = torch.device('cuda')
model  = ChestXrayModel().to(device)
model.load_state_dict(torch.load(
    '/media/morita/ubuntuHDD/chestxray/checkpoints/best_model.pth',
    map_location=device
))

loader = get_test_loader(img_size=224, batch_size=64)
record_result('DenseNet-121', model, loader, device,
              note='224px, 30epochs, baseline')
