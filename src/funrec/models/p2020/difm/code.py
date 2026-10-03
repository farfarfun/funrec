# -*- coding:utf-8 -*-
"""

Reference:
    [1] Lu W, Yu Y, Chang Y, et al. A Dual Input-aware Factorization Machine for CTR Prediction[C]//IJCAI. 2020: 3139-3145.(https://www.ijcai.org/Proceedings/2020/0434.pdf)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import SparseFeat, VarLenSparseFeat, combined_dnn_input
from funrec.layers import DNN, FM, InteractingLayer, concat_fun
from funrec.models.b2000 import BaseModel


class DIFM(BaseModel):
    """双重输入感知因子分解机（DIFM）网络架构。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        att_head_num: 多头自注意力网络中的头数。
        att_res: 是否在输出前使用标准残差连接。
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
        [1] Lu W, Yu Y, Chang Y, et al. A Dual Input-aware Factorization Machine for CTR Prediction[C]//IJCAI. 2020: 3139-3145.(https://www.ijcai.org/Proceedings/2020/0434.pdf)
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        att_head_num: int = 4,
        att_res: bool = True,
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
        super(DIFM, self).__init__(
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

        # InteractingLayer (used in AutoInt) = multi-head self-attention + Residual Network
        self.vector_wise_net = InteractingLayer(
            self.embedding_size, att_head_num, att_res, scaling=True, device=device
        )

        self.bit_wise_net = DNN(
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

        self.transform_matrix_P_vec = nn.Linear(
            self.sparse_feat_num * self.embedding_size, self.sparse_feat_num, bias=False
        ).to(device)
        self.transform_matrix_P_bit = nn.Linear(
            dnn_hidden_units[-1], self.sparse_feat_num, bias=False
        ).to(device)

        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.vector_wise_net.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.bit_wise_net.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(
            self.transform_matrix_P_vec.weight, l2=l2_reg_dnn
        )
        self.add_regularization_weight(
            self.transform_matrix_P_bit.weight, l2=l2_reg_dnn
        )

        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行向量级注意力、位级 DNN 与线性部分的前向计算并融合输出。"""
        sparse_embedding_list, _ = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        if not len(sparse_embedding_list) > 0:
            raise ValueError("there are no sparse features")

        att_input = concat_fun(sparse_embedding_list, axis=1)
        att_out = self.vector_wise_net(att_input)
        att_out = att_out.reshape(att_out.shape[0], -1)
        m_vec = self.transform_matrix_P_vec(att_out)

        dnn_input = combined_dnn_input(sparse_embedding_list, [])
        dnn_output = self.bit_wise_net(dnn_input)
        m_bit = self.transform_matrix_P_bit(dnn_output)

        m_x = m_vec + m_bit  # m_x is the complete input-aware factor

        logit = self.linear_model(X, sparse_feat_refine_weight=m_x)

        fm_input = torch.cat(sparse_embedding_list, dim=1)
        refined_fm_input = fm_input * m_x.unsqueeze(
            -1
        )  # \textbf{v}_{x,i}=m_{x,i} * \textbf{v}_i
        logit += self.fm(refined_fm_input)

        y_pred = self.out(logit)

        return y_pred
