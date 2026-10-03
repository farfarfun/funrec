# -*- coding:utf-8 -*-
"""
Reference:
    [1] Ruder S. An overview of multi-task learning in deep neural networks[J]. arXiv preprint arXiv:1706.05098, 2017.(https://arxiv.org/pdf/1706.05098.pdf)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.models.b2000 import BaseModel

from funrec.inputs import combined_dnn_input
from funrec.layers import DNN, PredictionLayer


class SharedBottom(BaseModel):
    """共享底层（Shared-Bottom）多任务学习网络架构。

    参数:
        dnn_feature_columns: 深度部分使用的特征列。
        bottom_dnn_hidden_units: 共享底层 DNN 各隐藏层的单元数，可为空列表。
        tower_dnn_hidden_units: 各任务独立塔（tower）DNN 各隐藏层的单元数，可为空列表。
        l2_reg_linear: 线性部分的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_dnn: DNN 的 L2 正则强度。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        dnn_activation: DNN 使用的激活函数。
        dnn_use_bn: DNN 激活前是否使用 BatchNormalization。
        task_types: 各任务的损失类型列表，``"binary"`` 对应二分类 logloss，
            ``"regression"`` 对应回归损失，例如 ``["binary", "regression"]``。
        task_names: 各任务的预测目标名称列表。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Ruder S. An overview of multi-task learning in deep neural networks[J]. arXiv preprint arXiv:1706.05098, 2017.(https://arxiv.org/pdf/1706.05098.pdf)
    """

    def __init__(
        self,
        dnn_feature_columns: list[Any],
        bottom_dnn_hidden_units: tuple[int, ...] = (256, 128),
        tower_dnn_hidden_units: tuple[int, ...] = (64,),
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
        super(SharedBottom, self).__init__(
            linear_feature_columns=[],
            dnn_feature_columns=dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            device=device,
            gpus=gpus,
        )
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ["binary", "regression"]:
                raise ValueError(
                    "task must be binary or regression, {} is illegal".format(task_type)
                )

        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.bottom_dnn_hidden_units = bottom_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        self.bottom_dnn = DNN(
            self.input_dim,
            bottom_dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            device=device,
        )
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [
                    DNN(
                        bottom_dnn_hidden_units[-1],
                        tower_dnn_hidden_units,
                        activation=dnn_activation,
                        dropout_rate=dnn_dropout,
                        use_bn=dnn_use_bn,
                        init_std=init_std,
                        device=device,
                    )
                    for _ in range(self.num_tasks)
                ]
            )
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.tower_dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        self.tower_dnn_final_layer = nn.ModuleList(
            [
                nn.Linear(
                    tower_dnn_hidden_units[-1]
                    if len(self.tower_dnn_hidden_units) > 0
                    else bottom_dnn_hidden_units[-1],
                    1,
                    bias=False,
                )
                for _ in range(self.num_tasks)
            ]
        )

        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.bottom_dnn.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.tower_dnn_final_layer.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行共享底层 DNN 与各任务独立塔的前向计算，返回各任务的预测结果拼接。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        shared_bottom_output = self.bottom_dnn(dnn_input)

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](shared_bottom_output)
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](shared_bottom_output)
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs
