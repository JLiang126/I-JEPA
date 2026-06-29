import torch 
import torch.nn as nn 

from utils.utils import PatchEmbedding, MultiHeadAttention, TransformerBlock

class IJEPA_ContextEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, dropout):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        
        # Without this the model is just passed a 'bag of patches'
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # Creates a tensor of the number of patches identifiers with each identifier having the number of embedding dimesions blank lines on it
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 3. The Assembly Line (Your utility)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)