import torch
from torch import nn

from timm.models.vision_transformer import Block
from models.ViTClassifier import ViTClassifier
from models.ViTEncoder import ViTEncoder

from torch_utils import get_2d_sincos_pos_embed

class ViTDecoder(nn.Module):
    # A simple transformer decoder with N transformer decoder stacks to recreate the image
    def __init__(self, embedding_dim, decoder_embedding_dim, num_patches, in_channels, patch_size, num_layers, num_heads, mlp_ratio):
        super().__init__()
        self.decoder_embed = nn.Linear(embedding_dim, decoder_embedding_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embedding_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embedding_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(num_layers)
            ])

        self.decoder_norm = nn.LayerNorm(decoder_embedding_dim)
        self.decoder_pred = nn.Linear(decoder_embedding_dim, patch_size[0]**2 * in_channels, bias=True)

        self.patch_size = patch_size


    def init_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches**0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patch_to_img(self, x):
        p = self.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h*p, h*p))
        return imgs

    def forward(self, x, ids_to_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_to_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_to_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]
    
        return x


class MaskedViTAutoencoder(nn.Module):
    # A class that creates a new ViT encoder and decoder, or loads it. It also creates a pat
    def __init__(self, embedding_dim, decoder_embedding_dim, patch_size, in_channels, decoder_num_layers, decoder_num_heads, decoder_mlp_ratio, encoder=None, img_size=None, encoder_num_heads=-1, encoder_num_layers=-1):
        super().__init__()

        self.encoder_embedding_dim=embedding_dim

        if encoder is None:
            self.encoder = ViTEncoder(img_size, patch_size, in_channels, embedding_dim, encoder_num_heads, encoder_num_layers, to_mask=True)
        else:
            self.encoder = encoder
            self.encoder.to_mask = True

        self.patch_size = patch_size
        self.num_patches = self.encoder.patcher.patch_embed.num_patches
        self.decoder = ViTDecoder(embedding_dim, decoder_embedding_dim, self.num_patches, in_channels, patch_size, decoder_num_layers, decoder_num_heads, decoder_mlp_ratio)

    def forward(self, img):
        latent, masked, ids_to_restore = self.encoder(img)
        pred_img = self.decoder(latent, ids_to_restore)

        return pred_img, masked, self.patch_size

    def toClassifier(self, num_classes, linear_hidden_dim=-1):
        self.encoder.set_mask(False)
        return ViTClassifier(self.encoder_embedding_dim, num_classes, linear_hidden_dim, self.encoder)
