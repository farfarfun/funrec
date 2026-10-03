import torch
import torch.nn as nn
import torch.nn.functional as F

from funrec.layers.core import LocalActivationUnit


class AttentionSequencePoolingLayer(nn.Module):
    """DIN 与 DIEN 中使用的注意力序列池化层。

    参数:
        att_hidden_units: 注意力网络各隐藏层的单元数。
        att_activation: 注意力网络使用的激活函数。
        weight_normalization: 是否对局部激活单元输出的注意力得分做归一化。
        return_score: 是否直接返回注意力得分而非加权求和结果。
        supports_masking: 若为 ``True``，输入需额外提供 mask。
        embedding_dim: 候选物品与行为序列的嵌入维度。

    参考文献:
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(
        self,
        att_hidden_units: tuple[int, ...] = (80, 40),
        att_activation: str = "sigmoid",
        weight_normalization: bool = False,
        return_score: bool = False,
        supports_masking: bool = False,
        embedding_dim: int = 4,
        **kwargs: object,
    ) -> None:
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.local_att = LocalActivationUnit(
            hidden_units=att_hidden_units,
            embedding_dim=embedding_dim,
            activation=att_activation,
            dropout_rate=0,
            use_bn=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        keys_length: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """基于注意力得分对用户行为序列做加权池化。

        输入形状:
            - ``query``: 3D 张量，形状为 ``(batch_size, 1, embedding_size)``。
            - ``keys``: 3D 张量，形状为 ``(batch_size, T, embedding_size)``。
            - ``keys_length``: 2D 张量，形状为 ``(batch_size, 1)``。

        输出形状:
            3D 张量，形状为 ``(batch_size, 1, embedding_size)``。
        """
        batch_size, max_length, _ = keys.size()

        # Mask
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking"
                )
            keys_masks = mask.unsqueeze(1)
        else:
            keys_masks = torch.arange(
                max_length, device=keys_length.device, dtype=keys_length.dtype
            ).repeat(batch_size, 1)  # [B, T]
            keys_masks = keys_masks < keys_length.view(-1, 1)  # 0, 1 mask
            keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]

        attention_score = self.local_att(query, keys)  # [B, T, 1]

        outputs = torch.transpose(attention_score, 1, 2)  # [B, 1, T]

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-(2**32) + 1)
        else:
            paddings = torch.zeros_like(outputs)

        outputs = torch.where(keys_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        # outputs = outputs / (keys.shape[-1] ** 0.05)

        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)  # [B, 1, T]

        if not self.return_score:
            # Weighted sum
            outputs = torch.matmul(outputs, keys)  # [B, 1, E]

        return outputs
