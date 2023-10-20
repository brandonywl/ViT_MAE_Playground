from models.PatchGenerator import PatchGenerator
from timm.models.vision_transformer import Block
from torch import nn

class ViTEncoder(nn.Module):
    # A simple transformer encoder that passes it through N transformer encoders. Also contains a patch encoder which can be swapped out with Instance Generator to encode over image segments
    def __init__(self, img_size, patch_size, in_channels, embedding_dim, num_head, num_layer, to_mask=False):
        super().__init__()
        self.patcher = PatchGenerator(img_size, patch_size, in_channels, embedding_dim, to_mask=to_mask)
        self.transformer = nn.Sequential(*[Block(embedding_dim, num_head) for _ in range(num_layer)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.to_mask = to_mask

    def set_mask(self, to_mask):
        self.to_mask = to_mask
        self.patcher.to_mask = to_mask


    def init_weight(self):
        self.apply(self._init_weight)

    def _init_weight(self, m):
        # Referencing https://github.com/facebookresearch/mae/blob/main/models_mae.py for MAE initialization
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patcher(x)

        if self.to_mask:
            x, mask, id_to_restore = x
            
        for blk in self.transformer:
            x = blk(x)
        x = self.norm(x)

        if self.to_mask:
            return x, mask, id_to_restore
        else:
            return x
