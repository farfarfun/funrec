# -*- coding:utf-8 -*-


from collections import OrderedDict, defaultdict
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from funutil import getLogger

from funrec.layers.sequence import SequencePoolingLayer
from funrec.layers.utils import concat_fun

logger = getLogger("funrec")
DEFAULT_GROUP_NAME = "default_group"


class SparseFeat:
    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        embedding_dim: int = 4,
        use_hash: bool = False,
        dtype: str = "int32",
        embedding_name: str = None,
        group_name: str = DEFAULT_GROUP_NAME,
    ):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            logger.info(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!"
            )
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim: int = embedding_dim
        self.use_hash: bool = use_hash
        self.dtype: str = dtype
        self.embedding_name = embedding_name
        self.group_name = group_name

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat:
    def __init__(
        self,
        sparsefeat: SparseFeat,
        maxlen: int,
        combiner: str = "mean",
        length_name: str = None,
    ):
        self.sparsefeat: SparseFeat = sparsefeat
        self.maxlen: int = maxlen
        self.combiner: str = combiner
        self.length_name: str = length_name

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat:
    def __init__(self, name, dimension=1, dtype="float32"):
        self.name = name
        self.dimension = dimension
        self.dtype = dtype

    def __hash__(self):
        return self.name.__hash__()


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


# def get_inputs_list(inputs):
#     return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def build_input_features(feature_columns):
    # Return OrderedDict: {feature_name:(start, start+dimension)}

    features = OrderedDict()

    start = 0
    for feat in feature_columns:
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
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def combined_dnn_input(sparse_embedding_list, dense_value_list):
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
    embedding_dict, features, feature_index, varlen_sparse_feature_columns, device
):
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
    feature_columns, init_std=0.0001, linear=False, sparse=False, device="cpu"
) -> nn.ModuleDict:
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
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
    X,
    sparse_embedding_dict,
    sparse_input_dict,
    sparse_feature_columns,
    return_feat_list=(),
    mask_feat_list=(),
    to_list=False,
):
    """
    Args:
        X: input Tensor [batch_size x hidden_dim]
        sparse_embedding_dict: nn.ModuleDict, {embedding_name: nn.Embedding}
        sparse_input_dict: OrderedDict, {feature_name:(start, start+dimension)}
        sparse_feature_columns: list, sparse features
        return_feat_list: list, names of feature to be returned, defualt () -> return all features
        mask_feat_list, list, names of feature to be masked in hash transform
    Return:
        group_embedding_dict: defaultdict(list)
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
    X, embedding_dict, sequence_input_dict, varlen_sparse_feature_columns
):
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


def get_dense_input(X, features, feature_columns):
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


def maxlen_lookup(X, sparse_input_dict, maxlen_column):
    if maxlen_column is None or len(maxlen_column) == 0:
        raise ValueError(
            "please add max length column for VarLenSparseFeat of DIN/DIEN input"
        )
    lookup_idx = np.array(sparse_input_dict[maxlen_column[0]])
    return X[:, lookup_idx[0] : lookup_idx[1]].long()
