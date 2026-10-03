import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


__all__ = [
    "AGRUCell",
    "AUGRUCell",
    "DynamicGRU",
]


class AGRUCell(nn.Module):
    """基于注意力的 GRU（AGRU）。

    参数:
        input_size: 输入特征维度。
        hidden_size: 隐藏状态维度。
        bias: 是否使用偏置项。

    参考文献:
        - Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter("weight_ih", self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter("weight_hh", self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter("bias_ih", self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter("bias_hh", self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor)
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(
        self, inputs: torch.Tensor, hx: torch.Tensor, att_score: torch.Tensor
    ) -> torch.Tensor:
        """用注意力得分替代更新门，计算当前时间步的隐藏状态。"""
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        hy = (1.0 - att_score) * hx + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    """带注意力更新门的 GRU（AUGRU）。

    参数:
        input_size: 输入特征维度。
        hidden_size: 隐藏状态维度。
        bias: 是否使用偏置项。

    参考文献:
        - Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter("weight_ih", self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter("weight_hh", self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter("bias_ih", self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter("bias_hh", self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor)
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(
        self, inputs: torch.Tensor, hx: torch.Tensor, att_score: torch.Tensor
    ) -> torch.Tensor:
        """用注意力得分缩放更新门，计算当前时间步的隐藏状态。"""
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1.0 - update_gate) * hx + update_gate * new_state
        return hy


class DynamicGRU(nn.Module):
    """支持 ``PackedSequence`` 输入、按注意力得分驱动的动态 GRU。

    参数:
        input_size: 输入特征维度。
        hidden_size: 隐藏状态维度。
        bias: 是否使用偏置项。
        gru_type: GRU 变体类型，``"AGRU"`` 或 ``"AUGRU"``。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        gru_type: str = "AGRU",
    ) -> None:
        super(DynamicGRU, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size

        if gru_type == "AGRU":
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == "AUGRU":
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(
        self,
        inputs: PackedSequence,
        att_scores: PackedSequence | None = None,
        hx: torch.Tensor | None = None,
    ) -> PackedSequence:
        """按时间步展开 ``PackedSequence``，逐步调用底层 GRU Cell。"""
        if not isinstance(inputs, PackedSequence) or not isinstance(
            att_scores, PackedSequence
        ):
            raise NotImplementedError(
                "DynamicGRU only supports packed input and att_scores"
            )

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(
                max_batch_size,
                self.hidden_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )

        outputs = torch.zeros(
            inputs.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device
        )

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                inputs[begin : begin + batch],
                hx[0:batch],
                att_scores[begin : begin + batch],
            )
            outputs[begin : begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)
