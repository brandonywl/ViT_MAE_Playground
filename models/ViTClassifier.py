from torch import nn
from models.ViTEncoder import ViTEncoder

class ViTClassifier(nn.Module):
    # A class that takes a trained ViT encoder and attaches a new Classifier Head after it
    def __init__(self, embedding_dim, num_classes, linear_hidden_dim=-1, encoder=None, img_size=None, patch_size=None, in_channels=None, num_head=None, num_layer=None, to_mask=False):
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ViTEncoder(img_size, patch_size, in_channels, embedding_dim, num_head, num_layer, to_mask=to_mask)

        self.classifier = ClassifierHead(embedding_dim, linear_hidden_dim, num_classes)

    def forward_loss(self, x, label):
        pass

    def forward(self, x):
        latent = self.encoder(x)
        return self.classifier(latent)
    
class ClassifierHead(nn.Module):
    # A simple classifier head with a fully connected layer before classification layer
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super().__init__()

        if hidden_dim == -1:
            layers = [
                nn.Linear(embedding_dim, num_classes)
            ]
        else:
            layers = [
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes)
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x = x.mean(dim=1)
        x = x[:, 0, :]
        return self.model(x)