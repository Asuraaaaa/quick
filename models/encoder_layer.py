import torch
from torch import nn

from .layer_norm import LayerNorm
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward import PositionwiseFeedForward
from .dca import DCA, CA

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # Add learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, d_model))  # 1000 is max seq len, adjust as needed

    def forward(self, x, src_mask=None):
        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]

        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class EncoderLayerDCA(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayerDCA, self).__init__()
        self.linear_ch1 = nn.Linear(63, d_model//2)
        self.linear_ch2 = nn.Linear(63, d_model//2)
        self.attention = DCA(d_model=d_model)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # Add learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, 63))  # 1000 is max seq len, adjust as needed

    def forward(self, x1, x2):
        # Add positional encoding
        # print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}, pos_embed shape: {self.pos_embed.shape}")
        x1 = x1 + self.pos_embed[:, :x1.size(1), :]
        x2 = x2 + self.pos_embed[:, :x2.size(1), :]

        # 1. compute self attention
        x1 = self.linear_ch1(x1)
        x2 = self.linear_ch2(x2)

        _x1, _x2 = x1, x2
        x = self.attention(x1, x2)
        
        # 2. add and norm
        x = self.dropout1(x)
        _x = torch.concat([_x1, _x2], dim=-1)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class EncoderLayerCA(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayerCA, self).__init__()
        self.linear_ch1 = nn.Linear(63, d_model//2)
        self.linear_ch2 = nn.Linear(63, d_model//2)
        self.attention = CA(q_dim=d_model//2, kv_dim=d_model//2, embed_dim=d_model)
        self.res_proj = nn.Linear(d_model//2, d_model)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # Add learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, 63))  # 1000 is max seq len, adjust as needed

    def forward(self, x1, x2):
        # Add positional encoding
        x1 = x1 + self.pos_embed[:, :x1.size(1), :]
        x2 = x2 + self.pos_embed[:, :x2.size(1), :]

        # 1. compute self attention
        x1 = self.linear_ch1(x1)
        x2 = self.linear_ch2(x2)

        _x1, _x2 = x1, x2
        x = self.attention(x1, x2)
        
        # 2. add and norm
        x = self.dropout1(x)
        _x = torch.concat([_x1, _x2], dim=-1)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

if __name__ == "__main__":
    # 简单测试
    attn = EncoderLayer(d_model=256, ffn_hidden=256, n_head=4, drop_prob=0.1)
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"Total parameters of attn: {total_params}")
    dcaatt = EncoderLayerDCA(d_model=256, ffn_hidden=256, n_head=4, drop_prob=0.1)
    total_params = sum(p.numel() for p in dcaatt.parameters())
    print(f"Total parameters of dcaatt: {total_params}")
    caatt = EncoderLayerCA(d_model=256, ffn_hidden=256, n_head=4, drop_prob=0.1)
    total_params = sum(p.numel() for p in caatt.parameters())
    print(f"Total parameters of caatt: {total_params}")

    x0 = torch.randn(2, 129, 63)
    x1 = torch.randn(2, 129, 63)
    x = torch.randn(2, 129, 256)
    dca_out = dcaatt(x0, x1)
    attn_out = attn(x,None)
    ca_out = caatt(x0, x1)
    print(f"Output shape: {attn_out.shape}")  # 应该是 (2, 63, 256)
    print(f"Output shape: {dca_out.shape}")  # 应该是 (2, 63, 256)
    print(f"Output shape: {ca_out.shape}")  # 应该是 (2, 63, 256)