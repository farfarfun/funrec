# -*- coding:utf-8 -*-
"""
Reference:
    [1] Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.
    (http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)

"""

from typing import Any

import torch
import torch.nn as nn

from funrec.layers.core import DNN
from funrec.layers.interaction import ConvLayer
from funrec.layers.utils import concat_fun
from funrec.models.b2000 import BaseModel


class CCPM(BaseModel):
    """卷积点击率预测模型（CCPM）。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度网络部分使用的特征列。
        conv_kernel_width: 各卷积层滤波器宽度组成的列表，可为空列表。
        conv_filters: 各卷积层滤波器数量组成的列表，可为空列表。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
        l2_reg_linear: 线性部分的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_dnn: DNN 的 L2 正则强度。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        dnn_use_bn: DNN 激活前是否使用 BatchNormalization。
        dnn_activation: DNN 使用的激活函数。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.
        (http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        conv_kernel_width: tuple[int, ...] = (6, 5),
        conv_filters: tuple[int, ...] = (4, 4),
        dnn_hidden_units: tuple[int, ...] = (256,),
        l2_reg_linear: float = 1e-5,
        l2_reg_embedding: float = 1e-5,
        l2_reg_dnn: float = 0,
        dnn_dropout: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: str = "binary",
        device: str = "cpu",
        dnn_use_bn: bool = False,
        dnn_activation: str = "relu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(CCPM, self).__init__(
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

        if len(conv_kernel_width) != len(conv_filters):
            raise ValueError(
                "conv_kernel_width must have same element with conv_filters"
            )

        filed_size = self.compute_input_dim(
            dnn_feature_columns, include_dense=False, feature_group=True
        )
        self.conv_layer = ConvLayer(
            field_size=filed_size,
            conv_kernel_width=conv_kernel_width,
            conv_filters=conv_filters,
            device=device,
        )
        self.dnn_input_dim = (
            self.conv_layer.filed_shape * self.embedding_size * conv_filters[-1]
        )
        self.dnn = DNN(
            self.dnn_input_dim,
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
        """执行线性部分与卷积-DNN 部分的前向计算并融合输出。"""
        linear_logit = self.linear_model(X)
        sparse_embedding_list, _ = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict, support_dense=False
        )
        if len(sparse_embedding_list) == 0:
            raise ValueError(
                "must have the embedding feature,now the embedding feature is None!"
            )
        conv_input = concat_fun(sparse_embedding_list, axis=1)
        conv_input_concact = torch.unsqueeze(conv_input, 1)
        pooling_result = self.conv_layer(conv_input_concact)
        flatten_result = pooling_result.view(pooling_result.size(0), -1)
        dnn_output = self.dnn(flatten_result)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = linear_logit + dnn_logit
        y_pred = self.out(logit)
        return y_pred
