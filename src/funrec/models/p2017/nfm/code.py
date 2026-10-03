# -*- coding:utf-8 -*-
"""
Reference:
    [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364. (https://arxiv.org/abs/1708.05027)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import combined_dnn_input
from funrec.layers import DNN, BiInteractionPooling
from funrec.models.b2000 import BaseModel


class NFM(BaseModel):
    """神经因子分解机（NFM）网络架构。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_linear: 线性部分的 L2 正则强度。
        l2_reg_dnn: DNN 的 L2 正则强度。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        bi_dropout: BiInteractionPooling 层输出的 dropout 比例。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        dnn_activation: DNN 使用的激活函数。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364. (https://arxiv.org/abs/1708.05027)
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        dnn_hidden_units: tuple[int, ...] = (128, 128),
        l2_reg_embedding: float = 1e-5,
        l2_reg_linear: float = 1e-5,
        l2_reg_dnn: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        bi_dropout: float = 0,
        dnn_dropout: float = 0,
        dnn_activation: str = "relu",
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(NFM, self).__init__(
            linear_feature_columns,
            dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
        )

        self.dnn = DNN(
            self.compute_input_dim(dnn_feature_columns, include_sparse=False)
            + self.embedding_size,
            dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=False,
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
        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = bi_dropout
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(bi_dropout)
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行线性部分、双线性交互池化与 DNN 部分的前向计算并融合输出。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        linear_logit = self.linear_model(X)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        bi_out = self.bi_pooling(fm_input)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)

        dnn_input = combined_dnn_input([bi_out], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        logit = linear_logit + dnn_logit

        y_pred = self.out(logit)

        return y_pred
