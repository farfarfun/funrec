# -*- coding:utf-8 -*-
"""
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""

from typing import Any

import torch
import torch.nn as nn

from funrec.inputs import (
    SparseFeat,
    VarLenSparseFeat,
    combined_dnn_input,
    embedding_lookup,
    get_varlen_pooling_list,
    maxlen_lookup,
    varlen_embedding_lookup,
)
from funrec.layers import DNN, AttentionSequencePoolingLayer
from funrec.models.b2000 import BaseModel


class DIN(BaseModel):
    """深度兴趣网络（DIN）架构。

    参数:
        dnn_feature_columns: 深度部分使用的特征列。
        history_feature_list: 需要作为历史行为序列处理的稀疏特征名列表。
        dnn_use_bn: DNN 激活前是否使用 BatchNormalization。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
        dnn_activation: DNN 使用的激活函数。
        att_hidden_size: 注意力网络各隐藏层的单元数。
        att_activation: 注意力网络使用的激活函数。
        att_weight_normalization: 是否对局部激活单元的注意力分数做归一化。
        l2_reg_dnn: DNN 的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        dnn_dropout: DNN 的 dropout 比例，取值范围 ``[0, 1)``。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(
        self,
        dnn_feature_columns: list[Any],
        history_feature_list: list[str],
        dnn_use_bn: bool = False,
        dnn_hidden_units: tuple[int, ...] = (256, 128),
        dnn_activation: str = "relu",
        att_hidden_size: tuple[int, ...] = (64, 16),
        att_activation: str = "Dice",
        att_weight_normalization: bool = False,
        l2_reg_dnn: float = 0.0,
        l2_reg_embedding: float = 1e-6,
        dnn_dropout: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(DIN, self).__init__(
            [],
            dnn_feature_columns,
            l2_reg_linear=0,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
        )

        self.sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
            if dnn_feature_columns
            else []
        )
        self.varlen_sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns))
            if dnn_feature_columns
            else []
        )

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()

        self.attention = AttentionSequencePoolingLayer(
            att_hidden_units=att_hidden_size,
            embedding_dim=att_emb_dim,
            att_activation=att_activation,
            return_score=False,
            supports_masking=False,
            weight_normalization=att_weight_normalization,
        )

        self.dnn = DNN(
            inputs_dim=self.compute_input_dim(dnn_feature_columns),
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            l2_reg=l2_reg_dnn,
            use_bn=dnn_use_bn,
        )
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行用户历史行为的注意力池化与 DNN 部分的前向计算。"""
        _, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )

        # sequence pooling part
        query_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_feature_columns,
            return_feat_list=self.history_feature_list,
            to_list=True,
        )
        keys_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.history_feature_columns,
            return_feat_list=self.history_fc_names,
            to_list=True,
        )
        dnn_input_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_feature_columns,
            to_list=True,
        )

        sequence_embed_dict = varlen_embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_varlen_feature_columns,
        )

        sequence_embed_list = get_varlen_pooling_list(
            sequence_embed_dict,
            X,
            self.feature_index,
            self.sparse_varlen_feature_columns,
            self.device,
        )

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)  # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [B, T, E]

        keys_length_feature_name = [
            feat.length_name
            for feat in self.varlen_sparse_feature_columns
            if feat.length_name is not None
        ]
        keys_length = torch.squeeze(
            maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1
        )  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)  # [B, 1, E]

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred

    def _compute_interest_dim(self) -> int:
        """计算历史行为特征拼接后的嵌入维度。"""
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim
