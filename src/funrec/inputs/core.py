# -*- coding:utf-8 -*-


from collections import OrderedDict, defaultdict
from itertools import chain
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from farlog import getLogger

from funrec.layers.sequence import SequencePoolingLayer
from funrec.layers.utils import concat_fun

logger = getLogger("funrec")
DEFAULT_GROUP_NAME = "default_group"


class SparseFeat:
    """描述一个离散（类别型）稀疏特征。

    参数:
        name: 特征名称。
        vocabulary_size: 该特征的取值个数（词表大小）。
        embedding_dim: 嵌入向量维度，传入 ``"auto"`` 时按词表大小自动推算。
        use_hash: 是否启用特征哈希（当前 PyTorch 版本暂不支持）。
        dtype: 特征的数据类型标识。
        embedding_name: 嵌入表名称，默认与 ``name`` 相同，用于多个特征共享嵌入表。
        group_name: 特征分组名称，用于按组聚合嵌入向量（如 FiBiNet）。
    """

    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        embedding_dim: int | str = 4,
        use_hash: bool = False,
        dtype: str = "int32",
        embedding_name: str | None = None,
        group_name: str = DEFAULT_GROUP_NAME,
    ) -> None:
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            logger.info(
                "注意：当前 PyTorch 版本暂不支持实时特征哈希，如需该能力请使用 TensorFlow 版本实现。"
            )
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim: int = embedding_dim
        self.use_hash: bool = use_hash
        self.dtype: str = dtype
        self.embedding_name = embedding_name
        self.group_name = group_name

    def __hash__(self) -> int:
        return self.name.__hash__()


class VarLenSparseFeat:
    """描述一个变长稀疏特征（如用户历史行为序列）。

    内部包装一个 ``SparseFeat`` 来复用其词表、嵌入维度等配置，
    并在此基础上补充序列长度与池化方式。

    参数:
        sparsefeat: 序列中单个元素对应的稀疏特征定义。
        maxlen: 序列的最大长度，超出部分会被截断，不足部分需在外部补 0。
        combiner: 序列池化方式，支持 ``"mean"``、``"sum"`` 等。
        length_name: 记录真实序列长度的特征名称；为 ``None`` 时按非零值个数推断。
    """

    def __init__(
        self,
        sparsefeat: SparseFeat,
        maxlen: int,
        combiner: str = "mean",
        length_name: str | None = None,
    ) -> None:
        self.sparsefeat: SparseFeat = sparsefeat
        self.maxlen: int = maxlen
        self.combiner: str = combiner
        self.length_name: str | None = length_name

    @property
    def name(self) -> str:
        """透传内部 ``SparseFeat`` 的特征名称。"""
        return self.sparsefeat.name

    @property
    def vocabulary_size(self) -> int:
        """透传内部 ``SparseFeat`` 的词表大小。"""
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self) -> int:
        """透传内部 ``SparseFeat`` 的嵌入维度。"""
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self) -> bool:
        """透传内部 ``SparseFeat`` 的哈希开关。"""
        return self.sparsefeat.use_hash

    @property
    def dtype(self) -> str:
        """透传内部 ``SparseFeat`` 的数据类型标识。"""
        return self.sparsefeat.dtype

    @property
    def embedding_name(self) -> str:
        """透传内部 ``SparseFeat`` 的嵌入表名称。"""
        return self.sparsefeat.embedding_name

    @property
    def group_name(self) -> str:
        """透传内部 ``SparseFeat`` 的分组名称。"""
        return self.sparsefeat.group_name

    def __hash__(self) -> int:
        return self.name.__hash__()


class DenseFeat:
    """描述一个连续数值特征。

    参数:
        name: 特征名称。
        dimension: 特征维度，默认 1（标量）。
        dtype: 特征的数据类型标识。
    """

    def __init__(self, name: str, dimension: int = 1, dtype: str = "float32") -> None:
        self.name = name
        self.dimension = dimension
        self.dtype = dtype

    def __hash__(self) -> int:
        return self.name.__hash__()


def get_feature_names(feature_columns: list[Any]) -> list[str]:
    """返回特征列对应的输入名称。"""
    features = build_input_features(feature_columns)
    return list(features.keys())


# def get_inputs_list(inputs):
#     return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def build_input_features(
    feature_columns: list[Any],
) -> OrderedDict[str, tuple[int, int]]:
    """构建特征名称到输入张量切片范围的映射。"""

    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        # 先校验特征列类型，避免对不支持的类型访问 `.name` 时抛出难以定位的 AttributeError
        if not isinstance(feat, (SparseFeat, DenseFeat, VarLenSparseFeat)):
            raise TypeError(f"不支持的特征列类型：{type(feat)!r}")
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
    return features


def combined_dnn_input(
    sparse_embedding_list: list[torch.Tensor], dense_value_list: list[torch.Tensor]
) -> torch.Tensor:
    """将稀疏特征嵌入和连续特征值拼接为深度网络输入。

    参数:
        sparse_embedding_list: 稀疏特征嵌入张量列表，每个元素形状为 ``(batch_size, 1, embedding_dim)``。
        dense_value_list: 连续特征值张量列表，每个元素形状为 ``(batch_size, dimension)``。

    返回:
        拼接后的二维张量，形状为 ``(batch_size, total_dim)``。

    异常:
        NotImplementedError: 两个列表都为空时没有可拼接的输入。
    """
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1
        )
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1
        )
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def get_varlen_pooling_list(
    embedding_dict: nn.ModuleDict,
    features: torch.Tensor,
    feature_index: "OrderedDict[str, tuple[int, int]]",
    varlen_sparse_feature_columns: list[VarLenSparseFeat],
    device: str,
) -> list[torch.Tensor]:
    """对每个变长稀疏特征做序列池化，得到定长的嵌入表示。

    参数:
        embedding_dict: 特征名称到嵌入模块的映射。
        features: 原始输入张量，形状为 ``(batch_size, total_dim)``。
        feature_index: 特征名称到输入张量切片范围的映射。
        varlen_sparse_feature_columns: 待处理的变长稀疏特征列列表。
        device: 运行设备。

    返回:
        每个变长特征池化后的嵌入张量列表。
    """
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.name]
        if feat.length_name is None:
            seq_mask = (
                features[
                    :, feature_index[feat.name][0] : feature_index[feat.name][1]
                ].long()
                != 0
            )

            emb = SequencePoolingLayer(
                mode=feat.combiner, supports_masking=True, device=device
            )([seq_emb, seq_mask])
        else:
            seq_length = features[
                :,
                feature_index[feat.length_name][0] : feature_index[feat.length_name][1],
            ].long()
            emb = SequencePoolingLayer(
                mode=feat.combiner, supports_masking=False, device=device
            )([seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


def create_embedding_matrix(
    feature_columns: list[Any],
    init_std: float = 0.0001,
    linear: bool = False,
    sparse: bool = False,
    device: str = "cpu",
) -> nn.ModuleDict:
    """为稀疏特征和变长稀疏特征创建共享的嵌入模块映射。

    参数:
        feature_columns: 特征列列表，仅 ``SparseFeat``/``VarLenSparseFeat`` 参与建表。
        init_std: 嵌入权重正态初始化的标准差。
        linear: 是否用于线性部分（嵌入维度固定为 1）。
        sparse: 是否使用稀疏梯度的 ``nn.Embedding``。
        device: 运行设备。

    返回:
        特征嵌入名称到 ``nn.Embedding`` 的模块字典。
    """
    sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
        if len(feature_columns)
        else []
    )

    varlen_sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
        if len(feature_columns)
        else []
    )

    embedding_dict = nn.ModuleDict(
        {
            feat.embedding_name: nn.Embedding(
                feat.vocabulary_size,
                feat.embedding_dim if not linear else 1,
                sparse=sparse,
            )
            for feat in sparse_feature_columns + varlen_sparse_feature_columns
        }
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


def embedding_lookup(
    X: torch.Tensor,
    sparse_embedding_dict: nn.ModuleDict,
    sparse_input_dict: "OrderedDict[str, tuple[int, int]]",
    sparse_feature_columns: list[SparseFeat],
    return_feat_list: tuple[str, ...] = (),
    mask_feat_list: tuple[str, ...] = (),
    to_list: bool = False,
) -> dict[str, list[torch.Tensor]] | list[torch.Tensor]:
    """按分组查找稀疏特征的嵌入向量。

    参数:
        X: 输入张量，形状为 ``(batch_size, hidden_dim)``。
        sparse_embedding_dict: 嵌入名称到 ``nn.Embedding`` 的映射。
        sparse_input_dict: 特征名称到输入张量切片范围的映射。
        sparse_feature_columns: 待查找的稀疏特征列列表。
        return_feat_list: 仅返回这些特征名称对应的嵌入；为空表示返回全部。
        mask_feat_list: 哈希转换中需要屏蔽的特征名称列表（当前哈希功能尚未实现）。
        to_list: 是否将按分组组织的结果展开为单一列表。

    返回:
        ``to_list`` 为 ``False`` 时返回分组名称到嵌入张量列表的映射；
        为 ``True`` 时返回展开后的嵌入张量列表。
    """
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list:
            # TODO: add hash function
            # if fc.use_hash:
            #     raise NotImplementedError("hash function is not implemented in this version!")
            lookup_idx = np.array(sparse_input_dict[feature_name])
            input_tensor = X[:, lookup_idx[0] : lookup_idx[1]].long()
            emb = sparse_embedding_dict[embedding_name](input_tensor)
            group_embedding_dict[fc.group_name].append(emb)
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(
    X: torch.Tensor,
    embedding_dict: nn.ModuleDict,
    sequence_input_dict: "OrderedDict[str, tuple[int, int]]",
    varlen_sparse_feature_columns: list[VarLenSparseFeat],
) -> dict[str, torch.Tensor]:
    """查找变长稀疏特征序列中每个位置对应的嵌入向量（未做池化）。

    参数:
        X: 输入张量，形状为 ``(batch_size, hidden_dim)``。
        embedding_dict: 特征名称到嵌入模块的映射。
        sequence_input_dict: 特征名称到输入张量切片范围的映射。
        varlen_sparse_feature_columns: 待查找的变长稀疏特征列列表。

    返回:
        特征名称到序列嵌入张量的映射，形状为 ``(batch_size, maxlen, embedding_dim)``。
    """
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            # lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
            # TODO: add hash function
            lookup_idx = sequence_input_dict[feature_name]
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](
            X[:, lookup_idx[0] : lookup_idx[1]].long()
        )  # (lookup_idx)

    return varlen_embedding_vec_dict


def get_dense_input(
    X: torch.Tensor,
    features: "OrderedDict[str, tuple[int, int]]",
    feature_columns: list[Any],
) -> list[torch.Tensor]:
    """从输入张量中切出所有连续特征的取值。

    参数:
        X: 输入张量，形状为 ``(batch_size, hidden_dim)``。
        features: 特征名称到输入张量切片范围的映射。
        feature_columns: 特征列列表，仅 ``DenseFeat`` 参与提取。

    返回:
        每个连续特征对应的张量列表，元素形状为 ``(batch_size, dimension)``。
    """
    dense_feature_columns = (
        list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
        if feature_columns
        else []
    )
    dense_input_list = []
    for fc in dense_feature_columns:
        lookup_idx = np.array(features[fc.name])
        input_tensor = X[:, lookup_idx[0] : lookup_idx[1]].float()
        dense_input_list.append(input_tensor)
    return dense_input_list


def maxlen_lookup(
    X: torch.Tensor,
    sparse_input_dict: "OrderedDict[str, tuple[int, int]]",
    maxlen_column: list[str] | None,
) -> torch.Tensor:
    """取出 DIN/DIEN 等模型所需的序列真实长度列。

    参数:
        X: 输入张量，形状为 ``(batch_size, hidden_dim)``。
        sparse_input_dict: 特征名称到输入张量切片范围的映射。
        maxlen_column: 记录序列长度的特征名称列表，取第一个元素使用。

    返回:
        序列长度张量。

    异常:
        ValueError: ``maxlen_column`` 为空。
    """
    if maxlen_column is None or len(maxlen_column) == 0:
        raise ValueError(
            "请为 DIN/DIEN 的 VarLenSparseFeat 输入补充最大长度（length_name）列"
        )
    lookup_idx = np.array(sparse_input_dict[maxlen_column[0]])
    return X[:, lookup_idx[0] : lookup_idx[1]].long()
