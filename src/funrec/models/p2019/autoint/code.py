# -*- coding:utf-8 -*-
"""

Reference:
    [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import combined_dnn_input
from funrec.layers import DNN, InteractingLayer, concat_fun
from funrec.models.b2000 import BaseModel


class AutoInt(BaseModel):
    """AutoInt 网络架构，基于多头自注意力自动学习特征交互。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        att_layer_num: InteractingLayer 的层数。
        att_head_num: 多头自注意力网络的头数。
        att_res: 是否在输出前使用标准残差连接。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
        dnn_activation: DNN 使用的激活函数。
        l2_reg_dnn: DNN 的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        dnn_use_bn: DNN 激活前是否使用 BatchNormalization。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        att_layer_num: int = 3,
        att_head_num: int = 2,
        att_res: bool = True,
        dnn_hidden_units: tuple[int, ...] = (256, 128),
        dnn_activation: str = "relu",
        l2_reg_dnn: float = 0,
        l2_reg_embedding: float = 1e-5,
        dnn_use_bn: bool = False,
        dnn_dropout: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(AutoInt, self).__init__(
            linear_feature_columns,
            dnn_feature_columns,
            l2_reg_linear=0,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
        )
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        field_num = len(self.embedding_dict)

        embedding_size = self.embedding_size

        if len(dnn_hidden_units) and att_layer_num > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size
        elif len(dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_num * embedding_size
        else:
            raise NotImplementedError

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
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
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        self.int_layers = nn.ModuleList(
            [
                InteractingLayer(embedding_size, att_head_num, att_res, device=device)
                for _ in range(att_layer_num)
            ]
        )

        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行线性部分、自注意力交互层与 DNN 部分的前向计算并融合输出。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        logit = self.linear_model(X)

        att_input = concat_fun(sparse_embedding_list, axis=1)

        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = torch.flatten(att_input, start_dim=1)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if (
            len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0
        ):  # Deep & Interacting Layer
            deep_out = self.dnn(dnn_input)
            stack_out = concat_fun([att_output, deep_out])
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:  # Error
            pass

        y_pred = self.out(logit)

        return y_pred
