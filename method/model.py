import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F


def number_parameters(Net, type_size=8):
    para = sum([np.prod(list(p.size())) for p in Net.parameters()])
    return para / 1024 * type_size / 1024


class Residual_Connection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Layer_Normal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MSA_Block(nn.Module):
    def __init__(self, dim_seq, num_heads, dim_head):
        super().__init__()
        dim_inner = dim_head * num_heads

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim_seq, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_seq)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        b, n_l, _, h = *x.shape, self.num_heads
        q, k, v = map(lambda t: rearrange(t, 'b nw_l (h d) -> b h nw_l d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h nw_l d -> b nw_l (h d)')
        out = self.to_out(out)

        return out


class transformer_block(nn.Module):
    def __init__(self, dim_seq, dim_mlp, num_heads, dim_head):
        super().__init__()

        self.attention_block = Residual_Connection(
            Layer_Normal(dim_seq, MSA_Block(dim_seq=dim_seq, num_heads=num_heads, dim_head=dim_head)))

        self.mlp_block = Residual_Connection(Layer_Normal(dim_seq, MLP_Block(dim=dim_seq, hidden_dim=dim_mlp)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)

        return x


class Transformer_Structure(nn.Module):
    def __init__(self, dim_seq=3456, num_heads=3, dim_head=18, num_encoders=8):
        super().__init__()

        self.order_embedding = nn.Parameter(torch.randn(1, 30, dim_seq))
        self.to_trans, self.to_seq = nn.Linear(dim_seq, dim_head), nn.Linear(dim_head, dim_seq)

        self.layers = nn.ModuleList([])
        for _ in range(num_encoders):
            self.layers.append(transformer_block(dim_seq=dim_head, num_heads=num_heads,
                                                 dim_mlp=dim_seq * 2, dim_head=dim_head)
                               )

    def forward(self, img):

        x = img + self.order_embedding
        x = self.to_trans(x)
        for layer in self.layers:
            x = layer(x)
        x = self.to_seq(x)
        return x


class Softmax_Classify(nn.Module):
    def __init__(self, hidden_size, num_linear, num_class):
        super().__init__()

        tmp_hidden_size = hidden_size

        self.layers = nn.ModuleList([])
        for _ in range(num_linear - 1):
            self.layers.append(nn.Linear(int(tmp_hidden_size), int(tmp_hidden_size / 2)))
            tmp_hidden_size /= 2

        self.layers.append(nn.Linear(int(tmp_hidden_size), num_class))

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        b, l, n = x.shape
        x = rearrange(x, 'b l n -> (b l) n')
        for layer in self.layers:
            x = layer(x)
        x = self.soft_max(x)
        x = rearrange(x, '(b l) n -> b l n', b=b)
        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Conv3d(nn.Module):
    def __init__(self, in_channels, num_levels=4, f_maps=16):
        super().__init__()

        self.in_channels = in_channels

        self.layers = nn.ModuleList([])
        for i in range(num_levels):
            self.layers.append(conv3x3x3(self.in_channels, f_maps * (2 ** i), stride=1))
            self.layers.append(nn.BatchNorm3d(f_maps * (2 ** i)))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=1))
            self.in_channels = f_maps * (2 ** i)

    def forward(self, x):
        b, l, c, n_l, n_h, n_w = x.shape

        x = rearrange(x, 'b l c n_l n_h n_w -> (b l) c n_l n_h n_w')

        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, '(b l) c n_l n_h n_w  -> b l (c n_l n_h n_w)', l=l)

        return x


class transformer_network(nn.Module):

    def __init__(self, *, in_channels=1, num_levels=4, f_maps=16, dim_hidden=3456, num_heads=3, dim_head=18,
                 num_encoders=8, num_linear=2, num_class=2):
        super().__init__()

        self._3dcnn = Conv3d(in_channels=in_channels, num_levels=num_levels, f_maps=f_maps)

        self.transformer_structure = Transformer_Structure(dim_seq=dim_hidden, num_heads=num_heads,
                                                           dim_head=dim_head, num_encoders=num_encoders)

        self.softmax_classify = Softmax_Classify(hidden_size=dim_hidden, num_linear=num_linear, num_class=num_class)

    def forward(self, img):

        x = rearrange(img, 'b (l c) n_l n_h n_w -> b l c n_l n_h n_w', c=1)
        x = self._3dcnn(x)
        x = self.transformer_structure(x)
        x = self.softmax_classify(x)

        return x


