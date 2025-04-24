import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset

class Net(nn.Module):
    def __init__(self, output_dim=128, intermediate_dim=256):
        super(Net, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.fc_in_features = self.backbone.fc.in_features
        self.backup_projection_head = None
        self.backbone.fc = nn.Identity()  
        self.projection_head = nn.Sequential(
            nn.Linear(self.fc_in_features, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, output_dim)
        ) 

    def freeze_basemodel_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def restore_backbone(self):
        if self.backup_projection_head is not None:
            self.projection_head = self.backup_projection_head
            self.backup_projection_head = None

    def linear_probe(self, class_dim):
        self.freeze_basemodel_encoder()
        self.backup_projection_head = (
            self.projection_head
        )
        self.projection_head = nn.Linear(self.fc_in_features, class_dim) 

    def forward(self, x):
        features = self.backbone(x)
        if self.projection_head is not None:
            features = self.projection_head(features)
        return features

class SimCLRDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = self.get_transform()

    def get_transform(self):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            color_distort,
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2


class SimCLR_Loss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLR_Loss, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, projections):
        projections_n =  torch.nn.functional.normalize(projections, dim=1)

        similarity_matrix = torch.matmul(projections_n, projections_n.T)

        similarity_matrix /= self.temperature

        batch_size = projections.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool, device=projections.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        labels = torch.cat([torch.arange(batch_size // 2) + (batch_size // 2),
                            torch.arange(batch_size // 2)], dim=0).to(projections.device)
        
        loss = self.criterion(similarity_matrix, labels)
        return loss

