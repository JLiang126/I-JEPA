import torch 
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be "

        self.num_heads = num_heads
        self.head_dim = embed_dim / num_heads # Should be fine given dimension matching

        self.scale = 1.0 / math.sqrt(self.head_dim) # Scaling factor for stable training stops exploding raw score which reduces vanishing gradient

        # remember y = Wx + b, creates Weights matrices and the bias vectors
        self.q_proj = nn.Linear(q_dim, embed_dim)
        self.k_proj = nn.Linear(k_dim, embed_dim)
        self.v_proj = nn.Linear(v_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q_in, k_in, v_in): # dimension, embedding dimension 
        batch, q_sequence = q_in.shape
        _, k_sequence = k_in.shape

        q = self.q_proj(q_in)
        k = self.k_proj(k_in)
        v = self.v_proj(v_in)

        # Reshape matrices, shape becomes: [Batch, Sequence, Heads, Head_Dim]
        q = q.view(batch, q_sequence, self.num_heads, self.head_dim)
        k = k.view(batch, k_sequence, self.num_heads, self.head_dim)
        v = v.view(batch, k_sequence, self.num_heads, self.head_dim)

        # Shape becomes: [Batch, Heads, Sequence, Head_Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attention = scores.softmax(dim=-1) # Shape: [Batch, Heads, q_sequence, k_sequence]

        context = attention @ v
        context = context.transpose(1, 2).contiguous() # .contiguous() aligns the memory
        context = context.view(batch, q_sequence, -1) # Squash back into [Batch, N_q, Embed_Dim]

        out = self.out_proj(context)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, activation=nn.GELU):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadAttention(embed_dim, embed_dim, embed_dim, embed_dim, num_heads)

        # FFN
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            activation(), 
            nn.Dropout(dropout), 
            
            nn.Linear(hidden_dim, embed_dim), 
            nn.Dropout(dropout))
        
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        if context is None: context = x

        norm_x = self.norm1(x)
        norm_context = self.norm1(context) if context is not x else norm_x

        attention_out = self.attention(q_in=norm_x, k_in=norm_context, v_in=norm_context) 
        x += self.residual_dropout(attention_out)

        mlp_out = self.mlp(self.norm2(x))
        x += self.residual_dropout(mlp_out)
