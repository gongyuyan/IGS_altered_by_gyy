import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor


class DenseGATConv(torch.nn.Module):
    r"""Dense version of GATConv (strictly following original GAT paper)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.debug = True

        # 线性映射
        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')

        # 👉 改为论文中的单一 attention 向量 a ∈ R^{2F'}
        self.att = Parameter(torch.empty(1, heads, 2 * out_channels))

        # bias
        if bias:
            if concat:
                self.bias = Parameter(torch.empty(heads * out_channels))
            else:
                self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, adj: Tensor, mask: OptTensor = None,
                add_loop: bool = True) -> Tensor:

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        # ====== 仅作为结构mask（论文：masked attention）======
        adj_mask = adj != 0  # (B, N, N)

        # ====== 线性映射 ======
        x = self.lin(x)  # (B, N, heads*out_channels)
        x = x.view(B, N, self.heads, self.out_channels)

        # ====== 构造 pair-wise 拼接 ======
        # x_i: (B, N, 1, heads, d) → (B, N, N, heads, d)
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1, -1)
        # x_j: (B, 1, N, heads, d) → (B, N, N, heads, d)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1, -1)

        # 拼接 [Wh_i || Wh_j]
        x_cat = torch.cat([x_i, x_j], dim=-1)  # (B, N, N, heads, 2d)

        # ====== attention logits ======
        alpha = (x_cat * self.att).sum(dim=-1)  # (B, N, N, heads)

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # ====== masked attention ======
        alpha = alpha.masked_fill(~adj_mask.unsqueeze(-1), -9e15)

        # softmax over neighbors j
        alpha = F.softmax(alpha, dim=2)

        # ====== 聚合 ======
        out = torch.einsum('bijn,bjhd->bihd', alpha, x)

        # ====== 多头处理 ======
        if self.concat:
            out = out.reshape(B, N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)

        # bias
        if self.bias is not None:
            out = out + self.bias

        # mask 节点
        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
