# -*- coding:utf-8 -*-
"""
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import combined_dnn_input
from funrec.layers import CIN, DNN
from funrec.models.b2000 import BaseModel


class XDeepFM(BaseModel):
    """xDeepFM 推荐模型。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
        cin_layer_size: 压缩交互网络（CIN）各隐藏层的特征图数量，可为空列表。
        cin_split_half: 为 ``True`` 时，每个隐藏层只有一半的特征图会连接到输出单元。
        cin_activation: CIN 特征图使用的激活函数。
        l2_reg_linear: 线性部分的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_dnn: DNN 的 L2 正则强度。
        l2_reg_cin: CIN 的 L2 正则强度。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        dnn_activation: DNN 使用的激活函数。
        dnn_use_bn: DNN 激活前是否使用 BatchNormalization。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        dnn_hidden_units: tuple[int, ...] = (256, 256),
        cin_layer_size: tuple[int, ...] = (
            256,
            128,
        ),
        cin_split_half: bool = True,
        cin_activation: str = "relu",
        l2_reg_linear: float = 0.00001,
        l2_reg_embedding: float = 0.00001,
        l2_reg_dnn: float = 0,
        l2_reg_cin: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        dnn_dropout: float = 0,
        dnn_activation: str = "relu",
        dnn_use_bn: bool = False,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(xDeepFM, self).__init__(
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
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
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

        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
            field_num = len(self.embedding_dict)
            if cin_split_half is True:
                self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            self.cin = CIN(
                field_num,
                cin_layer_size,
                cin_activation,
                cin_split_half,
                l2_reg_cin,
                seed,
                device=device,
            )
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: "weight" in x[0], self.cin.named_parameters()),
                l2=l2_reg_cin,
            )

        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行线性部分、CIN 部分与 DNN 部分的前向计算并融合输出。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )

        linear_logit = self.linear_model(X)

        final_logit = linear_logit  # linear
        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
            final_logit += cin_logit  # + CIN
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            final_logit += dnn_logit  # + Deep

        return self.out(final_logit)


# 兼容历史导入路径，同时让公开类名符合 PascalCase 规范。
xDeepFM = XDeepFM
