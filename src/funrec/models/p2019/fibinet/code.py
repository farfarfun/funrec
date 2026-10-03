# -*- coding:utf-8 -*-
"""

Reference:
    [1] Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import DenseFeat, SparseFeat, VarLenSparseFeat, combined_dnn_input
from funrec.layers import DNN, BilinearInteraction, SENETLayer
from funrec.models.b2000 import BaseModel


class FiBiNet(BaseModel):
    """特征重要性与双线性特征交互网络（FiBiNet）架构。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        bilinear_type: 双线性交互层使用的函数类型，可选 ``"all"``/``"each"``/``"interaction"``。
        reduction_ratio: SENET 层使用的压缩比，取值范围 ``[1, inf)``。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
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
        [1] Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        bilinear_type: str = "interaction",
        reduction_ratio: int = 3,
        dnn_hidden_units: tuple[int, ...] = (128, 128),
        l2_reg_linear: float = 1e-5,
        l2_reg_embedding: float = 1e-5,
        l2_reg_dnn: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        dnn_dropout: float = 0,
        dnn_activation: str = "relu",
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(FiBiNet, self).__init__(
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
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.field_size = len(self.embedding_dict)
        self.SE = SENETLayer(self.field_size, reduction_ratio, seed, device)
        self.Bilinear = BilinearInteraction(
            self.field_size, self.embedding_size, bilinear_type, seed, device
        )
        self.dnn = DNN(
            self.compute_input_dim(dnn_feature_columns),
            dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=False,
            init_std=init_std,
            device=device,
        )
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

    def compute_input_dim(
        self,
        feature_columns: list[Any],
        include_sparse: bool = True,
        include_dense: bool = True,
    ) -> int:
        """计算 FiBiNet 的 DNN 输入维度（双线性交互输出 + 可选稠密特征）。"""
        sparse_feature_columns = (
            list(
                filter(
                    lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)),
                    feature_columns,
                )
            )
            if len(feature_columns)
            else []
        )
        dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )
        field_size = len(sparse_feature_columns)

        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        embedding_size = sparse_feature_columns[0].embedding_dim
        sparse_input_dim = field_size * (field_size - 1) * embedding_size
        input_dim = 0

        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim

        return input_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行 SENET 特征重要性加权、双线性交互与 DNN 部分的前向计算并融合输出。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        sparse_embedding_input = torch.cat(sparse_embedding_list, dim=1)

        senet_output = self.SE(sparse_embedding_input)
        senet_bilinear_out = self.Bilinear(senet_output)
        bilinear_out = self.Bilinear(sparse_embedding_input)

        linear_logit = self.linear_model(X)
        temp = torch.split(
            torch.cat((senet_bilinear_out, bilinear_out), dim=1), 1, dim=1
        )
        dnn_input = combined_dnn_input(temp, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        if (
            len(self.linear_feature_columns) > 0 and len(self.dnn_feature_columns) > 0
        ):  # linear + dnn
            final_logit = linear_logit + dnn_logit
        elif len(self.linear_feature_columns) == 0:
            final_logit = dnn_logit
        elif len(self.dnn_feature_columns) == 0:
            final_logit = linear_logit
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)

        return y_pred
