# -*- coding:utf-8 -*-
"""
Reference:
    [1] Tang H, Liu J, Zhao M, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations[C]//Fourteenth ACM Conference on Recommender Systems. 2020.(https://dl.acm.org/doi/10.1145/3383313.3412236)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.models.b2000 import BaseModel

from funrec.inputs import combined_dnn_input
from funrec.layers import DNN, PredictionLayer


class PLE(BaseModel):
    """渐进式分层抽取（PLE）多任务学习架构，由多层定制门控（CGC）组成。

    参数:
        dnn_feature_columns: 深度部分使用的特征列。
        shared_expert_num: 任务共享专家的数量。
        specific_expert_num: 每个任务专属专家的数量。
        num_levels: CGC 层数。
        expert_dnn_hidden_units: 专家 DNN 各隐藏层的单元数，可为空列表。
        gate_dnn_hidden_units: 门控 DNN 各隐藏层的单元数，可为空列表。
        tower_dnn_hidden_units: 任务塔 DNN 各隐藏层的单元数，可为空列表。
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
        task_names: 各任务预测目标的名称列表。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Tang H, Liu J, Zhao M, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations[C]//Fourteenth ACM Conference on Recommender Systems. 2020.(https://dl.acm.org/doi/10.1145/3383313.3412236)
    """

    def __init__(
        self,
        dnn_feature_columns: list[Any],
        shared_expert_num: int = 1,
        specific_expert_num: int = 1,
        num_levels: int = 2,
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
        super(PLE, self).__init__(
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
            raise ValueError("num_tasks must be greater than 1!")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ["binary", "regression"]:
                raise ValueError(
                    "task must be binary or regression, {} is illegal".format(task_type)
                )

        self.specific_expert_num = specific_expert_num
        self.shared_expert_num = shared_expert_num
        self.num_levels = num_levels
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        def multi_module_list(
            num_level,
            num_tasks,
            expert_num,
            inputs_dim_level0,
            inputs_dim_not_level0,
            hidden_units,
        ):
            return nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.ModuleList(
                                [
                                    DNN(
                                        inputs_dim_level0
                                        if level_num == 0
                                        else inputs_dim_not_level0,
                                        hidden_units,
                                        activation=dnn_activation,
                                        l2_reg=l2_reg_dnn,
                                        dropout_rate=dnn_dropout,
                                        use_bn=dnn_use_bn,
                                        init_std=init_std,
                                        device=device,
                                    )
                                    for _ in range(expert_num)
                                ]
                            )
                            for _ in range(num_tasks)
                        ]
                    )
                    for level_num in range(num_level)
                ]
            )

        # 1. experts
        # task-specific experts
        self.specific_experts = multi_module_list(
            self.num_levels,
            self.num_tasks,
            self.specific_expert_num,
            self.input_dim,
            expert_dnn_hidden_units[-1],
            expert_dnn_hidden_units,
        )

        # shared experts
        self.shared_experts = multi_module_list(
            self.num_levels,
            1,
            self.specific_expert_num,
            self.input_dim,
            expert_dnn_hidden_units[-1],
            expert_dnn_hidden_units,
        )

        # 2. gates
        # gates for task-specific experts
        specific_gate_output_dim = self.specific_expert_num + self.shared_expert_num
        if len(gate_dnn_hidden_units) > 0:
            self.specific_gate_dnn = multi_module_list(
                self.num_levels,
                self.num_tasks,
                1,
                self.input_dim,
                expert_dnn_hidden_units[-1],
                gate_dnn_hidden_units,
            )
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.specific_gate_dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        self.specific_gate_dnn_final_layer = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(
                            gate_dnn_hidden_units[-1]
                            if len(gate_dnn_hidden_units) > 0
                            else self.input_dim
                            if level_num == 0
                            else expert_dnn_hidden_units[-1],
                            specific_gate_output_dim,
                            bias=False,
                        )
                        for _ in range(self.num_tasks)
                    ]
                )
                for level_num in range(self.num_levels)
            ]
        )

        # gates for shared experts
        shared_gate_output_dim = (
            self.num_tasks * self.specific_expert_num + self.shared_expert_num
        )
        if len(gate_dnn_hidden_units) > 0:
            self.shared_gate_dnn = nn.ModuleList(
                [
                    DNN(
                        self.input_dim
                        if level_num == 0
                        else expert_dnn_hidden_units[-1],
                        gate_dnn_hidden_units,
                        activation=dnn_activation,
                        l2_reg=l2_reg_dnn,
                        dropout_rate=dnn_dropout,
                        use_bn=dnn_use_bn,
                        init_std=init_std,
                        device=device,
                    )
                    for level_num in range(self.num_levels)
                ]
            )
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.shared_gate_dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        self.shared_gate_dnn_final_layer = nn.ModuleList(
            [
                nn.Linear(
                    gate_dnn_hidden_units[-1]
                    if len(gate_dnn_hidden_units) > 0
                    else self.input_dim
                    if level_num == 0
                    else expert_dnn_hidden_units[-1],
                    shared_gate_output_dim,
                    bias=False,
                )
                for level_num in range(self.num_levels)
            ]
        )

        # 3. tower dnn (task-specific)
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
            self.specific_experts,
            self.shared_experts,
            self.specific_gate_dnn_final_layer,
            self.shared_gate_dnn_final_layer,
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

    # a single cgc Layer
    def cgc_net(
        self, inputs: list[torch.Tensor], level_num: int
    ) -> list[torch.Tensor]:
        """执行单层定制门控（CGC），输出各任务专属表示与共享表示。"""
        # inputs: [task1, task2, ... taskn, shared task]

        # 1. experts
        # task-specific experts
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                specific_expert_output = self.specific_experts[level_num][i][j](
                    inputs[i]
                )
                specific_expert_outputs.append(specific_expert_output)

        # shared experts
        shared_expert_outputs = []
        for k in range(self.shared_expert_num):
            shared_expert_output = self.shared_experts[level_num][0][k](inputs[-1])
            shared_expert_outputs.append(shared_expert_output)

        # 2. gates
        # gates for task-specific experts
        cgc_outs = []
        for i in range(self.num_tasks):
            # concat task-specific expert and task-shared expert
            cur_experts_outputs = (
                specific_expert_outputs[
                    i * self.specific_expert_num : (i + 1) * self.specific_expert_num
                ]
                + shared_expert_outputs
            )
            cur_experts_outputs = torch.stack(cur_experts_outputs, 1)

            # gate dnn
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.specific_gate_dnn[level_num][i][0](inputs[i])
                gate_dnn_out = self.specific_gate_dnn_final_layer[level_num][i](
                    gate_dnn_out
                )
            else:
                gate_dnn_out = self.specific_gate_dnn_final_layer[level_num][i](
                    inputs[i]
                )
            gate_mul_expert = torch.matmul(
                gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs
            )  # (bs, 1, dim)
            cgc_outs.append(gate_mul_expert.squeeze(1))

        # gates for shared experts
        cur_experts_outputs = specific_expert_outputs + shared_expert_outputs
        cur_experts_outputs = torch.stack(cur_experts_outputs, 1)

        if len(self.gate_dnn_hidden_units) > 0:
            gate_dnn_out = self.shared_gate_dnn[level_num](inputs[-1])
            gate_dnn_out = self.shared_gate_dnn_final_layer[level_num](gate_dnn_out)
        else:
            gate_dnn_out = self.shared_gate_dnn_final_layer[level_num](inputs[-1])
        gate_mul_expert = torch.matmul(
            gate_dnn_out.softmax(1).unsqueeze(1), cur_experts_outputs
        )  # (bs, 1, dim)
        cgc_outs.append(gate_mul_expert.squeeze())

        return cgc_outs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """逐层执行 CGC 抽取，并通过各任务塔输出多任务预测结果。"""
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # repeat `dnn_input` for several times to generate cgc input
        ple_inputs = [dnn_input] * (
            self.num_tasks + 1
        )  # [task1, task2, ... taskn, shared task]
        ple_outputs = []
        for i in range(self.num_levels):
            ple_outputs = self.cgc_net(inputs=ple_inputs, level_num=i)
            ple_inputs = ple_outputs

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](ple_outputs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](ple_outputs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs
