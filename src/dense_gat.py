import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptTensor


class DenseGATConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, heads * out_channels,
                          bias=False, weight_initializer='glorot')

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if bias:
            if concat:
                self.bias = Parameter(torch.Tensor(heads * out_channels))
            else:
                self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: OptTensor = None,
        add_loop: bool = True,
    ) -> Tensor:

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, device=adj.device)
            adj[:, idx, idx] = 1

        # 线性变换
        x = self.lin(x)  # (B, N, heads*out_channels)
        x = x.view(B, N, self.heads, self.out_channels)

        # 计算 attention logits
        alpha_src = (x * self.att_src).sum(dim=-1)  # (B, N, heads)
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # (B, N, heads)

        alpha = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)
        # shape: (B, N, N, heads)

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # 使用 dense adjacency 作为 mask
        # 关键：adj 参与计算图，保证 saliency 可求
        alpha = alpha * adj.unsqueeze(-1)

        # softmax over neighbors
        alpha = torch.softmax(alpha, dim=2)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 消息聚合
        out = torch.einsum('bijn,bjnh->binh', alpha, x)
        # (B, N, heads, out_channels)

        if self.concat:
            out = out.reshape(B, N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'{self.in_channels}, {self.out_channels}, heads={self.heads})'
        )
