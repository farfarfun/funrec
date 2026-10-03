# -*- coding:utf-8 -*-
from typing import Any

import torch
import torch.nn as nn
from torch.nn import Module


class Dice(Module):
    """DIN 中使用的数据自适应激活函数，可视为 PReLU 的推广，能根据输入数据的分布自适应地调整校正点。

    输入形状:
        - 2 维: ``[batch_size, embedding_size(features)]``
        - 3 维: ``[batch_size, num_features, embedding_size(features)]``

    输出形状:
        与输入形状相同。

    参数:
        emb_size: 输入的嵌入维度。
        dim: 输入张量的维度，仅支持 2 或 3。
        epsilon: BatchNorm 的数值稳定项。
        device: 运行设备。

    参考文献:
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(
        self, emb_size: int, dim: int = 2, epsilon: float = 1e-8, device: str = "cpu"
    ) -> None:
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据输入分布自适应地计算校正后的激活值。"""
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


class Identity(Module):
    """恒等映射层，原样返回输入，用于在激活函数位置占位。"""

    def __init__(self, **kwargs: Any) -> None:
        super(Identity, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """原样返回输入。"""
        return inputs


def activation_layer(
    act_name: str | type[nn.Module], hidden_size: int | None = None, dice_dim: int = 2
) -> Module:
    """根据名称或类型构造激活层。

    参数:
        act_name: 激活函数名称（字符串）或 ``nn.Module`` 子类。
        hidden_size: 使用 Dice 激活函数时所需的维度。
        dice_dim: 使用 Dice 激活函数时的维度参数。

    返回:
        构造好的激活层实例。

    异常:
        NotImplementedError: 当 ``act_name`` 不是已知的激活函数名称或 ``nn.Module`` 子类时抛出。
    """
    if isinstance(act_name, str):
        if act_name.lower() == "sigmoid":
            return nn.Sigmoid()
        elif act_name.lower() == "linear":
            return Identity()
        elif act_name.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif act_name.lower() == "dice":
            assert dice_dim
            return Dice(hidden_size, dice_dim)
        elif act_name.lower() == "prelu":
            return nn.PReLU()
    elif issubclass(act_name, nn.Module):
        return act_name()
    raise NotImplementedError
