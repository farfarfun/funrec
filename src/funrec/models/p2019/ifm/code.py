# -*- coding:utf-8 -*-
"""

Reference:
    [1] Yu Y, Wang Z, Yuan B. An Input-aware Factorization Machine for Sparse Prediction[C]//IJCAI. 2019: 1466-1472.(https://www.ijcai.org/Proceedings/2019/0203.pdf)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import SparseFeat, VarLenSparseFeat, combined_dnn_input
from funrec.layers import DNN, FM
from funrec.models.b2000 import BaseModel


class IFM(BaseModel):
    """输入感知因子分解机（IFM）网络架构。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
        l2_reg_linear: 线性部分的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_dnn: DNN 的 L2 正则强度。
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
        [1] Yu Y, Wang Z, Yuan B. An Input-aware Factorization Machine for Sparse Prediction[C]//IJCAI. 2019: 1466-1472.(https://www.ijcai.org/Proceedings/2019/0203.pdf)
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
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
    ) -> None:
        super(IFM, self).__init__(
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

        if not len(dnn_hidden_units) > 0:
            raise ValueError("dnn_hidden_units is null!")

        self.fm = FM()

        self.factor_estimating_net = DNN(
            self.compute_input_dim(dnn_feature_columns, include_dense=False),
            dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            device=device,
        )
        self.sparse_feat_num = len(
            list(
                filter(
                    lambda x: (
                        isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat)
                    ),
                    dnn_feature_columns,
                )
            )
        )
        self.transform_weight_matrix_P = nn.Linear(
            dnn_hidden_units[-1], self.sparse_feat_num, bias=False
        ).to(device)

        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.factor_estimating_net.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(
            self.transform_weight_matrix_P.weight, l2=l2_reg_dnn
        )

        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行输入感知因子估计、FM 交互与线性部分的前向计算并融合输出。"""
        sparse_embedding_list, _ = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        if not len(sparse_embedding_list) > 0:
            raise ValueError("there are no sparse features")

        dnn_input = combined_dnn_input(
            sparse_embedding_list, []
        )  # (batch_size, feat_num * embedding_size)
        dnn_output = self.factor_estimating_net(dnn_input)
        dnn_output = self.transform_weight_matrix_P(dnn_output)  # m'_{x}
        input_aware_factor = self.sparse_feat_num * dnn_output.softmax(
            1
        )  # input_aware_factor m_{x,i}

        logit = self.linear_model(X, sparse_feat_refine_weight=input_aware_factor)

        fm_input = torch.cat(sparse_embedding_list, dim=1)
        refined_fm_input = fm_input * input_aware_factor.unsqueeze(
            -1
        )  # \textbf{v}_{x,i}=m_{x,i}\textbf{v}_i
        logit += self.fm(refined_fm_input)

        y_pred = self.out(logit)

        return y_pred
