import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import activation_layer

__all__ = ["LocalActivationUnit", "DNN", "PredictionLayer", "Conv2dSame"]


class LocalActivationUnit(nn.Module):
    """DIN 中使用的局部激活单元，根据候选物品自适应地刻画用户兴趣表示。

    输入形状:
        两个 3D 张量组成的列表，形状分别为 ``(batch_size, 1, embedding_size)`` 和
        ``(batch_size, T, embedding_size)``。

    输出形状:
        3D 张量，形状为 ``(batch_size, T, 1)``。

    参数:
        hidden_units: 注意力网络各隐藏层的单元数。
        embedding_dim: 候选物品与行为序列的嵌入维度。
        activation: 注意力网络使用的激活函数。
        dropout_rate: 注意力网络输出的 dropout 比例，取值范围 ``[0, 1)``。
        dice_dim: 使用 Dice 激活函数时的维度参数。
        l2_reg: 注意力网络核权重矩阵的 L2 正则强度。
        use_bn: 激活前是否对注意力网络使用 BatchNormalization。

    参考文献:
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (64, 32),
        embedding_dim: int = 4,
        activation: str = "sigmoid",
        dropout_rate: float = 0,
        dice_dim: int = 3,
        l2_reg: float = 0,
        use_bn: bool = False,
    ) -> None:
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(
            inputs_dim=4 * embedding_dim,
            hidden_units=hidden_units,
            activation=activation,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            dice_dim=dice_dim,
            use_bn=use_bn,
        )

        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query: torch.Tensor, user_behavior: torch.Tensor) -> torch.Tensor:
        """计算候选物品与用户行为序列各位置的注意力得分。

        参数:
            query: 候选物品嵌入，形状为 ``(batch_size, 1, embedding_size)``。
            user_behavior: 用户历史行为序列嵌入，形状为 ``(batch_size, T, embedding_size)``。

        返回:
            注意力得分，形状为 ``(batch_size, T, 1)``。
        """
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavior_len = user_behavior.size(1)

        queries = query.expand(-1, user_behavior_len, -1)

        attention_input = torch.cat(
            [queries, user_behavior, queries - user_behavior, queries * user_behavior],
            dim=-1,
        )  # as the source code, subtraction simulates verctors' difference
        attention_output = self.dnn(attention_input)

        attention_score = self.dense(attention_output)  # [B, T, 1]

        return attention_score


class DNN(nn.Module):
    """多层感知机（MLP）。

    输入形状:
        nD 张量，形状为 ``(batch_size, ..., input_dim)``，最常见的是 2D 输入
        ``(batch_size, input_dim)``。

    输出形状:
        nD 张量，形状为 ``(batch_size, ..., hidden_size[-1])``；以 2D 输入为例，
        输出形状为 ``(batch_size, hidden_size[-1])``。

    参数:
        inputs_dim: 输入特征维度。
        hidden_units: 各隐藏层的单元数，决定层数。
        activation: 使用的激活函数。
        l2_reg: 核权重矩阵的 L2 正则强度，取值范围 ``[0, 1)``。
        dropout_rate: dropout 比例，取值范围 ``[0, 1)``。
        use_bn: 激活前是否使用 BatchNormalization。
        init_std: 权重正态初始化的标准差。
        dice_dim: 使用 Dice 激活函数时的维度参数。
        seed: 随机种子。
        device: 运行设备。
    """

    def __init__(
        self,
        inputs_dim: int,
        hidden_units: tuple[int, ...] | list[int],
        activation: str = "relu",
        l2_reg: float = 0,
        dropout_rate: float = 0,
        use_bn: bool = False,
        init_std: float = 0.0001,
        dice_dim: int = 3,
        seed: int = 1024,
        device: str = "cpu",
    ) -> None:
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [
                nn.Linear(hidden_units[i], hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ]
        )

        if self.use_bn:
            self.bn = nn.ModuleList(
                [
                    nn.BatchNorm1d(hidden_units[i + 1])
                    for i in range(len(hidden_units) - 1)
                ]
            )

        self.activation_layers = nn.ModuleList(
            [
                activation_layer(activation, hidden_units[i + 1], dice_dim)
                for i in range(len(hidden_units) - 1)
            ]
        )

        for name, tensor in self.linears.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """逐层执行线性变换、BatchNormalization、激活和 dropout。"""
        deep_input = inputs

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class PredictionLayer(nn.Module):
    """根据任务类型将网络输出映射为最终预测值。

    参数:
        task: 任务类型，``"binary"`` 对应二分类（sigmoid），``"regression"`` 对应回归，
            ``"multiclass"`` 对应多分类（不做额外变换）。
        use_bias: 是否额外加一个可学习的偏置项。
    """

    def __init__(self, task: str = "binary", use_bias: bool = True, **kwargs: Any) -> None:
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """按任务类型对输入 logit 做偏置和激活变换。"""
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output


class Conv2dSame(nn.Conv2d):
    """类似 TensorFlow ``"SAME"`` 填充方式的 2D 卷积包装器。

    参数:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        kernel_size: 卷积核尺寸。
        stride: 卷积步长。
        padding: 占位参数，实际填充量由 ``forward`` 动态计算，传入值不生效。
        dilation: 卷积膨胀系数。
        groups: 分组卷积的组数。
        bias: 是否使用偏置项。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按 ``"SAME"`` 方式动态填充输入后执行卷积。"""
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        out = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return out
