# -*- coding:utf-8 -*-
"""
Reference:
    [1] Ma X, Zhao L, Huang G, et al. Entire space multi-task model: An effective approach for estimating post-click conversion rate[C]//The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.(https://dl.acm.org/doi/10.1145/3209978.3210104)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import combined_dnn_input
from funrec.layers import DNN
from funrec.models.b2000 import BaseModel


class ESMM(BaseModel):
    """全空间多任务模型（ESMM）架构。

    参数:
        dnn_feature_columns: 深度部分使用的特征列。
        tower_dnn_hidden_units: 各任务独立塔（tower）DNN 各隐藏层的单元数，可为空列表。
        l2_reg_linear: 线性部分的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_dnn: DNN 的 L2 正则强度。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        dnn_activation: DNN 使用的激活函数。
        dnn_use_bn: DNN 激活前是否使用 BatchNormalization。
        task_types: 各任务的损失类型列表，ESMM 要求均为 ``"binary"``，
            例如 ``["binary", "binary"]``。
        task_names: 各任务的预测目标名称列表，长度必须为 2。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Ma X, Zhao L, Huang G, et al. Entire space multi-task model: An effective approach for estimating post-click conversion rate[C]//The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.(https://dl.acm.org/doi/10.1145/3209978.3210104)
    """

    def __init__(
        self,
        dnn_feature_columns: list[Any],
        tower_dnn_hidden_units: tuple[int, ...] = (256, 128),
        l2_reg_linear: float = 0.00001,
        l2_reg_embedding: float = 0.00001,
        l2_reg_dnn: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        dnn_dropout: float = 0,
        dnn_activation: str = "relu",
        dnn_use_bn: bool = False,
        task_types: tuple[str, ...] = ("binary", "binary"),
        task_names: tuple[str, ...] = ("ctr", "ctcvr"),
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(ESMM, self).__init__(
            linear_feature_columns=[],
            dnn_feature_columns=dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task="binary",
            device=device,
            gpus=gpus,
        )
        self.num_tasks = len(task_names)
        if self.num_tasks != 2:
            raise ValueError("the length of task_names must be equal to 2")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type != "binary":
                raise ValueError(
                    "task must be binary in ESMM, {} is illegal".format(task_type)
                )

        input_dim = self.compute_input_dim(dnn_feature_columns)

        self.ctr_dnn = DNN(
            input_dim,
            tower_dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            device=device,
        )
        self.cvr_dnn = DNN(
            input_dim,
            tower_dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            device=device,
        )

        self.ctr_dnn_final_layer = nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)
        self.cvr_dnn_final_layer = nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)

        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.ctr_dnn.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.cvr_dnn.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(self.ctr_dnn_final_layer.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.cvr_dnn_final_layer.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """分别计算 CTR、CVR 塔的预测结果，并按 ``CTCVR = CTR * CVR`` 融合输出。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        ctr_output = self.ctr_dnn(dnn_input)
        cvr_output = self.cvr_dnn(dnn_input)

        ctr_logit = self.ctr_dnn_final_layer(ctr_output)
        cvr_logit = self.cvr_dnn_final_layer(cvr_output)

        ctr_pred = self.out(ctr_logit)
        cvr_pred = self.out(cvr_logit)

        ctcvr_pred = ctr_pred * cvr_pred  # CTCVR = CTR * CVR

        task_outs = torch.cat([ctr_pred, ctcvr_pred], -1)
        return task_outs
