import torch
import torch.nn as nn


__all__ = ["SequencePoolingLayer", "KMaxPooling"]


class SequencePoolingLayer(nn.Module):
    """对变长序列特征/多值特征执行池化操作（sum、mean 或 max）。

    输入形状:
        由两个张量组成的列表 ``[seq_value, seq_len]``：

        - ``seq_value``: 3D 张量，形状为 ``(batch_size, T, embedding_size)``。
        - ``seq_len``: 2D 张量，形状为 ``(batch_size, 1)``，表示每个序列的有效长度。

    输出形状:
        3D 张量，形状为 ``(batch_size, 1, embedding_size)``。

    参数:
        mode: 池化方式，可选 ``"sum"``/``"mean"``/``"max"``。
        supports_masking: 若为 ``True``，输入需额外提供 mask。
        device: 运行设备。
    """

    def __init__(
        self, mode: str = "mean", supports_masking: bool = False, device: str = "cpu"
    ) -> None:
        super(SequencePoolingLayer, self).__init__()
        if mode not in ["sum", "mean", "max"]:
            raise ValueError("parameter mode should in [sum, mean, max]")
        self.supports_masking = supports_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)

    def _sequence_mask(
        self, lengths: torch.Tensor, maxlen: int | None = None, dtype: torch.dtype = torch.bool
    ) -> torch.Tensor:
        """返回标记每个序列前 N 个有效位置的 mask 张量。"""
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list: list[torch.Tensor]) -> torch.Tensor:
        """按配置的池化方式聚合变长序列的嵌入表示。"""
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list  # [B, T, E], [B, 1]
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = (
                seq_value_len_list  # [B, T, E], [B, 1]
            )
            mask = self._sequence_mask(
                user_behavior_length,
                maxlen=uiseq_embed_list.shape[1],
                dtype=torch.float32,
            )  # [B, 1, maxlen]
            mask = torch.transpose(mask, 1, 2)  # [B, maxlen, 1]

        embedding_size = uiseq_embed_list.shape[-1]

        mask = torch.repeat_interleave(mask, embedding_size, dim=2)  # [B, maxlen, E]

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = uiseq_embed_list * mask.float()
        hist = torch.sum(hist, dim=1, keepdim=False)

        if self.mode == "mean":
            self.eps = self.eps.to(user_behavior_length.device)
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)

        hist = torch.unsqueeze(hist, dim=1)
        return hist


class KMaxPooling(nn.Module):
    """沿指定维度选取前 k 个最大值的 K-Max 池化层。

    输入形状:
        nD 张量，形状为 ``(batch_size, ..., input_dim)``。

    输出形状:
        nD 张量，形状为 ``(batch_size, ..., output_dim)``。

    参数:
        k: 在 ``axis`` 维度上取 top-k 的 k 值。
        axis: 执行 top-k 操作的维度。
        device: 运行设备。
    """

    def __init__(self, k: int, axis: int, device: str = "cpu") -> None:
        super(KMaxPooling, self).__init__()
        self.k: int = k
        self.axis: int = axis
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """沿 ``axis`` 维度返回前 ``k`` 个最大值。"""
        if self.axis < 0 or self.axis >= len(inputs.shape):
            raise ValueError(
                f"axis must be 0~{len(inputs.shape) - 1},now is {self.axis}"
            )

        if self.k < 1 or self.k > inputs.shape[self.axis]:
            raise ValueError(
                "k must be in 1 ~ %d,now k is %d" % (inputs.shape[self.axis], self.k)
            )

        out = torch.topk(inputs, k=self.k, dim=self.axis, sorted=True)[0]
        return out
