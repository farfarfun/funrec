# -*- coding:utf-8 -*-
"""
Reference:
    [1] Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.(https://arxiv.org/abs/1704.05194)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import build_input_features
from funrec.layers import PredictionLayer
from funrec.models.b2000 import BaseModel, Linear


class MLR(BaseModel):
    """混合逻辑回归 / 分段线性模型（MLR）。

    参数:
        region_feature_columns: 分区（region）部分使用的特征列。
        base_feature_columns: 基础（base）部分使用的特征列；为 ``None`` 时与
            ``region_feature_columns`` 相同。
        bias_feature_columns: 偏置（bias）部分使用的特征列。
        region_num: 分段数量，需大于 1。
        l2_reg_linear: 线性权重的 L2 正则强度。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.(https://arxiv.org/abs/1704.05194)
    """

    def __init__(
        self,
        region_feature_columns: list[Any],
        base_feature_columns: list[Any] | None = None,
        bias_feature_columns: list[Any] | None = None,
        region_num: int = 4,
        l2_reg_linear: float = 1e-5,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(MLR, self).__init__(
            region_feature_columns,
            region_feature_columns,
            task=task,
            device=device,
            gpus=gpus,
        )

        if region_num <= 1:
            raise ValueError("region_num must > 1")

        self.l2_reg_linear = l2_reg_linear
        self.init_std = init_std
        self.seed = seed
        self.device = device

        self.region_num = region_num
        self.region_feature_columns = region_feature_columns
        self.base_feature_columns = base_feature_columns
        self.bias_feature_columns = bias_feature_columns

        if base_feature_columns is None or len(base_feature_columns) == 0:
            self.base_feature_columns = region_feature_columns

        if bias_feature_columns is None:
            self.bias_feature_columns = []

        self.feature_index = build_input_features(
            self.region_feature_columns
            + self.base_feature_columns
            + self.bias_feature_columns
        )

        self.region_linear_model = nn.ModuleList(
            [
                Linear(
                    self.region_feature_columns,
                    self.feature_index,
                    self.init_std,
                    self.device,
                )
                for i in range(self.region_num)
            ]
        )

        self.base_linear_model = nn.ModuleList(
            [
                Linear(
                    self.base_feature_columns,
                    self.feature_index,
                    self.init_std,
                    self.device,
                )
                for i in range(self.region_num)
            ]
        )

        if self.bias_feature_columns is not None and len(self.bias_feature_columns) > 0:
            self.bias_model = nn.Sequential(
                Linear(
                    self.bias_feature_columns,
                    self.feature_index,
                    self.init_std,
                    self.device,
                ),
                PredictionLayer(task="binary", use_bias=False),
            )

        self.prediction_layer = PredictionLayer(task=task, use_bias=False)

        self.to(self.device)

    def get_region_score(
        self, inputs: torch.Tensor, region_number: int
    ) -> torch.Tensor:
        """计算各分段的 softmax 权重分数。"""
        region_logit = torch.cat(
            [self.region_linear_model[i](inputs) for i in range(region_number)], dim=-1
        )
        region_score = nn.Softmax(dim=-1)(region_logit)
        return region_score

    def get_learner_score(
        self, inputs: torch.Tensor, region_number: int
    ) -> torch.Tensor:
        """计算各分段线性模型的预测分数。"""
        learner_score = self.prediction_layer(
            torch.cat(
                [self.region_linear_model[i](inputs) for i in range(region_number)],
                dim=-1,
            )
        )
        return learner_score

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """融合各分段权重与对应子模型预测，得到最终输出。"""
        region_score = self.get_region_score(X, self.region_num)
        learner_score = self.get_learner_score(X, self.region_num)

        final_logit = torch.sum(region_score * learner_score, dim=-1, keepdim=True)

        if self.bias_feature_columns is not None and len(self.bias_feature_columns) > 0:
            bias_score = self.bias_model(X)
            final_logit = final_logit * bias_score
        return final_logit
