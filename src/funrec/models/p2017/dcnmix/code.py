# -*- coding:utf-8 -*-
"""
Reference:
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)

    [2] Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020. (https://arxiv.org/abs/2008.13535)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import combined_dnn_input
from funrec.layers import DNN, CrossNetMix
from funrec.models.b2000 import BaseModel


class DCNMix(BaseModel):
    """DCN-Mix 模型，使用低秩混合专家形式的 Cross 网络。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        cross_num: Cross 网络的层数，正整数。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_cross: Cross 网络的 L2 正则强度。
        l2_reg_dnn: DNN 的 L2 正则强度。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        low_rank: 低秩空间的维度，正整数。
        num_experts: 专家数量，正整数。
        dnn_activation: DNN 使用的激活函数。
        dnn_use_bn: DNN 激活前是否使用 BatchNormalization。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)

        [2] Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020. (https://arxiv.org/abs/2008.13535)
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        cross_num: int = 2,
        dnn_hidden_units: tuple[int, ...] = (128, 128),
        l2_reg_linear: float = 0.00001,
        l2_reg_embedding: float = 0.00001,
        l2_reg_cross: float = 0.00001,
        l2_reg_dnn: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        dnn_dropout: float = 0,
        low_rank: int = 32,
        num_experts: int = 4,
        dnn_activation: str = "relu",
        dnn_use_bn: bool = False,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(DCNMix, self).__init__(
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
        )
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        self.dnn = DNN(
            self.compute_input_dim(dnn_feature_columns),
            dnn_hidden_units,
            activation=dnn_activation,
            use_bn=dnn_use_bn,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            init_std=init_std,
            device=device,
        )
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = (
                self.compute_input_dim(dnn_feature_columns) + dnn_hidden_units[-1]
            )
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns)

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.crossnet = CrossNetMix(
            in_features=self.compute_input_dim(dnn_feature_columns),
            low_rank=low_rank,
            num_experts=num_experts,
            layer_num=cross_num,
            device=device,
        )
        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.dnn.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)
        regularization_modules = [
            self.crossnet.U_list,
            self.crossnet.V_list,
            self.crossnet.C_list,
        ]
        for module in regularization_modules:
            self.add_regularization_weight(module, l2=l2_reg_cross)

        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行线性部分、低秩混合 Cross 网络与 DNN 部分的前向计算并融合输出。"""
        logit = self.linear_model(X)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:  # Deep & Cross
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((cross_out, deep_out), dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.cross_num > 0:  # Only Cross
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:  # Error
            pass
        y_pred = self.out(logit)
        return y_pred
