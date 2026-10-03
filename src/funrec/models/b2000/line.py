# -*- coding:utf-8 -*-

from typing import Any

import torch
import torch.nn as nn


from funrec.inputs import (
    DenseFeat,
    SparseFeat,
    VarLenSparseFeat,
    create_embedding_matrix,
    get_varlen_pooling_list,
    varlen_embedding_lookup,
)


class Linear(nn.Module):
    """线性部分模块，汇总稀疏特征的一阶嵌入与稠密特征的线性加权结果。

    参数:
        feature_columns: 参与线性计算的特征列。
        feature_index: 特征名到其在拼接输入张量中起止列下标的映射。
        init_std: 嵌入向量初始化的标准差。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
    """

    def __init__(
        self,
        feature_columns: list[Any],
        feature_index: dict[str, tuple[int, int]],
        init_std: float = 0.0001,
        device: str = "cpu",
    ) -> None:
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )
        self.dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        self.varlen_sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        self.embedding_dict = create_embedding_matrix(
            feature_columns, init_std, linear=True, sparse=False, device=device
        )

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(
                torch.Tensor(
                    sum(fc.dimension for fc in self.dense_feature_columns), 1
                ).to(device)
            )
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(
        self,
        X: torch.Tensor,
        sparse_feat_refine_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算线性部分的输出 logit，可选地对稀疏特征一阶权重做输入感知修正。"""
        sparse_embedding_list = [
            self.embedding_dict[feat.embedding_name](
                X[
                    :,
                    self.feature_index[feat.name][0] : self.feature_index[feat.name][1],
                ].long()
            )
            for feat in self.sparse_feature_columns
        ]

        dense_value_list = [
            X[:, self.feature_index[feat.name][0] : self.feature_index[feat.name][1]]
            for feat in self.dense_feature_columns
        ]

        sequence_embed_dict = varlen_embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.varlen_sparse_feature_columns,
        )
        varlen_embedding_list = get_varlen_pooling_list(
            sequence_embed_dict,
            X,
            self.feature_index,
            self.varlen_sparse_feature_columns,
            self.device,
        )

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = (
                    sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
                )
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit
