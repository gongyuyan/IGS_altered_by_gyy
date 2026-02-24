import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor


class DenseGATConv(torch.nn.Module):
    r"""Dense version of GATConv."""

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

        # 线性映射：F -> heads * out_channels
        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')

        # attention 参数 a = [a_l || a_r]
        self.att_l = Parameter(torch.empty(1, heads, out_channels))
        self.att_r = Parameter(torch.empty(1, heads, out_channels))

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
        glorot(self.att_l)
        glorot(self.att_r)
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

        # ====== DEBUG 1 ======
        if self.training and self.debug:
            print("adj min/max:", adj.min().item(), adj.max().item())
            print("adj zero count:", (adj == 0).sum().item())
            self.debug = False

        # 线性映射
        x = self.lin(x)  # (B, N, heads*out_channels)
        x = x.view(B, N, self.heads, self.out_channels)

        # 计算 attention score
        alpha_l = (x * self.att_l).sum(dim=-1)  # (B, N, heads)
        alpha_r = (x * self.att_r).sum(dim=-1)  # (B, N, heads)

        # broadcast 到 (B, N, N, heads)
        alpha = alpha_l.unsqueeze(2) + alpha_r.unsqueeze(1)

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # ====== DEBUG 2 ======
        if self.training and self.debug:
            print("alpha(before mask) mean/std:",
                  alpha.mean().item(),
                  alpha.std().item())
            self.debug = False

        # 只在存在边的位置计算 softmax
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        alpha = F.softmax(alpha, dim=2)

        # ====== DEBUG 3 ======
        if self.training and self.debug:
            print("alpha(after softmax) mean/std:",
                  alpha.mean().item(),
                  alpha.std().item())
            print("alpha min/max:",
                  alpha.min().item(),
                  alpha.max().item())
            self.debug = False

        # 消息聚合
        out = torch.einsum('bijn,bjhd->bihd', alpha, x)

        # ====== DEBUG 4 ======
        if self.training and self.debug:
            print("out std:", out.std().item())
            self.debug = False

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
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
