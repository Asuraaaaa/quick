from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class DCA(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.sub_d_model = int(d_model / 2)
        self.d_model = d_model
        # 两个通道各自独立线性层
        self.qkv0 = nn.Linear(self.sub_d_model, self.sub_d_model *3, bias=True)        
        self.qkv1 = nn.Linear(self.sub_d_model, self.sub_d_model *3, bias=True)
        # self.attn_drop = nn.Dropout(0.0)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x0, x1):

        B, L, _ = x0.shape

        # ===== 生成 QKV =====
        qkv0 = self.qkv0(x0)  # [B, L, 128]
        qkv1 = self.qkv1(x1)

        Q0, K0, V0 = qkv0.chunk(3, dim=-1)
        Q1, K1, V1 = qkv1.chunk(3, dim=-1)

        # ===== 每组切成2个head =====
        # [B, L, 128] -> [B, 2, L, 64]
        def split_head(x):
            return x.view(B, L, 2, -1).transpose(1, 2)

        Q0 = split_head(Q0)
        K0 = split_head(K0)
        V0 = split_head(V0)

        Q1 = split_head(Q1)
        K1 = split_head(K1)
        V1 = split_head(V1)

        # ===== 头级重组 =====
        # head0: Q00-K10
        # head1: Q10-K00
        # head2: Q01-K01
        # head3: Q11-K11

        Q = torch.cat([
            Q0[:, 0:1],  # Q00
            Q0[:, 1:2],  # Q01
            Q1[:, 0:1],  # Q10
            Q1[:, 1:2],  # Q11
        ], dim=1)

        K = torch.cat([
            K1[:, 0:1],  # K10
            K0[:, 1:2],  # K01
            K0[:, 0:1],  # K00
            K1[:, 1:2],  # K11
        ], dim=1)

        V = torch.cat([
            V1[:, 0:1],  # V10
            V0[:, 1:2],  # V01
            V0[:, 0:1],  # V00
            V1[:, 1:2],  # V11
        ], dim=1)

        # Q = torch.cat([Q0, Q1], dim=1)
        # K = torch.cat([K1, K0], dim=1)
        # V = torch.cat([V1, V0], dim=1)

        # ===== 一次性 attention 的参数量 =====

        # attn01 = torch.matmul(Q0, K1.transpose(-2, -1)) / (64 ** 0.5)
        # attn01 = F.softmax(attn01, dim=-1)
        # out01 = torch.matmul(attn01, V1)

        # attn10 = torch.matmul(Q1, K0.transpose(-2, -1)) / (64 ** 0.5)
        # attn10 = F.softmax(attn10, dim=-1)
        # out10 = torch.matmul(attn10, V0)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (64 ** 0.5)
        attn = F.softmax(attn, dim=-1)
        # attn = self.attn_drop(attn)
        out = torch.matmul(attn, V)  # [B,4,L,64]
        # out = 0.5*(out01 + out10)  # [B,2,L,64]
        # ===== 合并 head =====
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        return self.out_proj(out)

class CA(nn.Module):
    def __init__(
        self,
        q_dim: int = 128,
        kv_dim: int = 128,
        embed_dim: int = 256,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q 来自 x_q
        self.q_proj = nn.Linear(q_dim, embed_dim, bias=bias)
        # K, V 来自 x_kv
        self.k_proj = nn.Linear(kv_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(kv_dim, embed_dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor):
        B, Nq, _ = x_q.shape
        B2, Nk, _ = x_kv.shape
        assert B == B2, "x_q 和 x_kv 的 batch size 必须一致"

        # 线性映射
        q = self.q_proj(x_q)   # [B, Nq, embed_dim]
        k = self.k_proj(x_kv)  # [B, Nk, embed_dim]
        v = self.v_proj(x_kv)  # [B, Nk, embed_dim]

        # 拆成多头
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, Nq, d]
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, Nk, d]
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, Nk, d]

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, Nq, Nk]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        out = torch.matmul(attn, v)  # [B, h, Nq, d]

        # 拼接多头
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.embed_dim)  # [B, Nq, embed_dim]

        # 输出映射
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out
    
if __name__ == "__main__":
    # 简单测试
    dca = DCA(d_model=256)
    ca = CA()
    x0 = torch.randn(2, 129, 128)
    x1 = torch.randn(2, 129, 128)
    out = dca(x0, x1)
    print(f"DCA output shape: {out.shape}")  # 应该是 (2, 129, 256)

    ca = CA(q_dim=128, kv_dim=128, embed_dim=256, num_heads=4)
    out_ca = ca(x0, x1)
    print(f"CA output shape: {out_ca.shape}")  # 应该是 (2, 129, 256)