import torch
import torch.nn as nn
from torchvision.models.video import mc3_18

class IntegrationModel(nn.Module):
    def __init__(self, ae_dim, vae_dim, convnext_dim, swin_dim, mc3_dim, num_classes=2):
        super(IntegrationModel, self).__init__()

        # Define the fusion layers to integrate features
        self.fusion_layers = nn.Sequential(
          nn.Linear(ae_dim + vae_dim + convnext_dim + swin_dim + mc3_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)  # Output layer for classification
        )

    def forward(self, ae_features, vae_features, convnext_features, swin_features, mc3_features):
        
        combined_features = torch.cat((ae_features, vae_features, convnext_features, swin_features, mc3_features), dim=1)

        output = self.fusion_layers(combined_features)

        return output

ae_dim = 128
vae_dim = 64
convnext_dim = 512
swin_dim = 768
mc3_dim = 1024


integration_model = IntegrationModel(ae_dim, vae_dim, convnext_dim, swin_dim, mc3_dim)

# Example usage:

ae_features = torch.randn(1, ae_dim)
vae_features = torch.randn(1, vae_dim)
convnext_features = torch.randn(1, convnext_dim)
swin_features = torch.randn(1, swin_dim)
mc3_features = torch.randn(1, mc3_dim)

# Perform forward pass through the integration model
output = integration_model(ae_features, vae_features, convnext_features, swin_features, mc3_features)
print("Output:", output)
