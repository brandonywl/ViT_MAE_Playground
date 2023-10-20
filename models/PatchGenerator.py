from timm.models.vision_transformer import PatchEmbed
import torch
from torch import nn
from torch_utils import get_2d_sincos_pos_embed

class PatchGenerator(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dims, to_mask=False):
        super().__init__()

        self.to_mask = to_mask

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dims) # Use Conv2d instead to get the patches
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims), requires_grad=False)  # fixed sin-cos embedding

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


    def random_masking(self, x, mask_ratio):
        """
        Reference https://github.com/facebookresearch/mae/blob/main/models_mae.py
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.75):
        x = self.patch_embed(imgs)

        # Apply pos embedding
        x = x + self.pos_embed[:, 1:, :]

        if self.to_mask:
            # Apply masking
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.to_mask:
            return x, mask, ids_restore

        else:
            return x