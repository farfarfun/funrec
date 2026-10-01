"""
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import combined_dnn_input
from funrec.layers import DNN, FM
from funrec.models.b2000 import BaseModel


class DeepFM(BaseModel):
    """实现同时组合线性、因子分解机和深度网络的 DeepFM 模型。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度网络部分使用的特征列。
        use_fm: 是否启用因子分解机部分。
        dnn_hidden_units: 深度网络各隐藏层的单元数。
        task: 任务类型，支持 ``binary`` 和 ``regression``。
        device: 运行设备，例如 ``cpu`` 或 ``cuda:0``。
        gpus: 用于并行计算的 GPU 编号或设备列表。

    返回:
        初始化后的 PyTorch DeepFM 模型。
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        use_fm: bool = True,
        dnn_hidden_units: tuple[int, ...] = (256, 128),
        l2_reg_linear: float = 0.00001,
        l2_reg_embedding: float = 0.00001,
        l2_reg_dnn: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        dnn_dropout: float = 0,
        dnn_activation: str = "relu",
        dnn_use_bn: bool = False,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(DeepFM, self).__init__(
            linear_feature_columns,
            dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
            *args,
            **kwargs,
        )

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(
                self.compute_input_dim(dnn_feature_columns),
                dnn_hidden_units,
                activation=dnn_activation,
                l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn,
                init_std=init_std,
                device=device,
            )
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """根据批量特征张量计算预测结果。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred
