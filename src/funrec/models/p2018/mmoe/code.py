# -*- coding:utf-8 -*-
"""
Reference:
    [1] Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C] (https://dl.acm.org/doi/10.1145/3219819.3220007)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import combined_dnn_input
from funrec.layers import DNN, PredictionLayer
from funrec.models.b2000 import BaseModel


class MMOE(BaseModel):
    """多门混合专家模型（MMOE）架构。

    参数:
        dnn_feature_columns: 深度部分使用的特征列。
        num_experts: 专家数量，需大于 1。
        expert_dnn_hidden_units: 专家 DNN 各隐藏层的单元数，可为空列表。
        gate_dnn_hidden_units: 门控 DNN 各隐藏层的单元数，可为空列表。
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
        [1] Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C] (https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(
        self,
        dnn_feature_columns: list[Any],
        num_experts: int = 3,
        expert_dnn_hidden_units: tuple[int, ...] = (256, 128),
        gate_dnn_hidden_units: tuple[int, ...] = (64,),
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
        super(MMOE, self).__init__(
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
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ["binary", "regression"]:
                raise ValueError(
                    "task must be binary or regression, {} is illegal".format(task_type)
                )

        self.num_experts = num_experts
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        # expert dnn
        self.expert_dnn = nn.ModuleList(
            [
                DNN(
                    self.input_dim,
                    expert_dnn_hidden_units,
                    activation=dnn_activation,
                    l2_reg=l2_reg_dnn,
                    dropout_rate=dnn_dropout,
                    use_bn=dnn_use_bn,
                    init_std=init_std,
                    device=device,
                )
                for _ in range(self.num_experts)
            ]
        )

        # gate dnn
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList(
                [
                    DNN(
                        self.input_dim,
                        gate_dnn_hidden_units,
                        activation=dnn_activation,
                        l2_reg=l2_reg_dnn,
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
                    self.gate_dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        self.gate_dnn_final_layer = nn.ModuleList(
            [
                nn.Linear(
                    gate_dnn_hidden_units[-1]
                    if len(gate_dnn_hidden_units) > 0
                    else self.input_dim,
                    self.num_experts,
                    bias=False,
                )
                for _ in range(self.num_tasks)
            ]
        )

        # tower dnn (task-specific)
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [
                    DNN(
                        expert_dnn_hidden_units[-1],
                        tower_dnn_hidden_units,
                        activation=dnn_activation,
                        l2_reg=l2_reg_dnn,
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
                    if len(tower_dnn_hidden_units) > 0
                    else expert_dnn_hidden_units[-1],
                    1,
                    bias=False,
                )
                for _ in range(self.num_tasks)
            ]
        )

        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        regularization_modules = [
            self.expert_dnn,
            self.gate_dnn_final_layer,
            self.tower_dnn_final_layer,
        ]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    module.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行专家网络、门控加权与各任务塔的前向计算，返回各任务预测结果拼接。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # expert dnn
        expert_outs = []
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)  # (bs, num_experts, dim)

        # gate dnn
        mmoe_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(
                gate_dnn_out.softmax(1).unsqueeze(1), expert_outs
            )  # (bs, 1, dim)
            mmoe_outs.append(gate_mul_expert.squeeze(1))

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](mmoe_outs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs
