# -*- coding:utf-8 -*-
"""


Reference:
    [1] Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
         Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.layers import DNN, LogTransformLayer
from funrec.models.b2000 import BaseModel


class AFN(BaseModel):
    """自适应阶数因子分解网络（AFN）架构。

    说明: 为保持模型接口一致性，这里仅提供 AFN 的非集成版本；集成版本 AFN+
    请参考原作者的 PyTorch 实现（DeepCTR-Torch）或 TensorFlow 实现（AFN-AAAI-20）。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        ltl_hidden_size: 对数变换层（LogTransformLayer）中对数神经元的数量。
        afn_dnn_hidden_units: AFN 中 DNN 各隐藏层的单元数，可为空列表。
        l2_reg_linear: 线性部分的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_dnn: DNN 的 L2 正则强度。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        dnn_activation: DNN 使用的激活函数。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
             Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        ltl_hidden_size: int = 256,
        afn_dnn_hidden_units: tuple[int, ...] = (256, 128),
        l2_reg_linear: float = 0.00001,
        l2_reg_embedding: float = 0.00001,
        l2_reg_dnn: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        dnn_dropout: float = 0,
        dnn_activation: str = "relu",
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(AFN, self).__init__(
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

        self.ltl = LogTransformLayer(
            len(self.embedding_dict), self.embedding_size, ltl_hidden_size
        )
        self.afn_dnn = DNN(
            self.embedding_size * ltl_hidden_size,
            afn_dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=True,
            init_std=init_std,
            device=device,
        )
        self.afn_dnn_linear = nn.Linear(afn_dnn_hidden_units[-1], 1)
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行线性部分与对数变换-DNN 部分的前向计算并融合输出。"""
        sparse_embedding_list, _ = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        logit = self.linear_model(X)
        if len(sparse_embedding_list) == 0:
            raise ValueError(
                "Sparse embeddings not provided. AFN only accepts sparse embeddings as input."
            )

        afn_input = torch.cat(sparse_embedding_list, dim=1)
        ltl_result = self.ltl(afn_input)
        afn_logit = self.afn_dnn(ltl_result)
        afn_logit = self.afn_dnn_linear(afn_logit)

        logit += afn_logit
        y_pred = self.out(logit)

        return y_pred
