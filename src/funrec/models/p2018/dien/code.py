"""


Reference:
    [1] Zhou G, Mou N, Fan Y, et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018. (https://arxiv.org/pdf/1809.03672.pdf)
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from funrec.inputs import (
    DenseFeat,
    SparseFeat,
    VarLenSparseFeat,
    combined_dnn_input,
    concat_fun,
    embedding_lookup,
    get_dense_input,
    maxlen_lookup,
)
from funrec.layers import DNN, AttentionSequencePoolingLayer, DynamicGRU
from funrec.models.b2000 import BaseModel


class DIEN(BaseModel):
    """深度兴趣演化网络（DIEN）架构。

    参数:
        dnn_feature_columns: 深度部分使用的特征列。
        history_feature_list: 需要作为历史行为序列处理的稀疏特征名列表。
        gru_type: 兴趣演化层使用的 GRU 变体，可选 ``"GRU"``/``"AIGRU"``/``"AGRU"``/``"AUGRU"``。
        use_negsampling: 是否使用负采样计算辅助损失。
        alpha: 辅助损失（auxiliary loss）的权重。
        use_bn: DNN 激活前是否使用 BatchNormalization。
        dnn_hidden_units: DNN 各隐藏层的单元数，可为空列表。
        dnn_activation: DNN 使用的激活函数。
        att_hidden_units: 注意力网络各隐藏层的单元数。
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
        [1] Zhou G, Mou N, Fan Y, et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018. (https://arxiv.org/pdf/1809.03672.pdf)
    """

    def __init__(
        self,
        dnn_feature_columns: list[Any],
        history_feature_list: list[str],
        gru_type: str = "GRU",
        use_negsampling: bool = False,
        alpha: float = 1.0,
        use_bn: bool = False,
        dnn_hidden_units: tuple[int, ...] = (256, 128),
        dnn_activation: str = "relu",
        att_hidden_units: tuple[int, ...] = (64, 16),
        att_activation: str = "relu",
        att_weight_normalization: bool = True,
        l2_reg_dnn: float = 0,
        l2_reg_embedding: float = 1e-6,
        dnn_dropout: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(DIEN, self).__init__(
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

        self.item_features = history_feature_list
        self.use_negsampling = use_negsampling
        self.alpha = alpha
        self._split_columns()

        # structure: embedding layer -> interest extractor layer -> interest evolution layer -> DNN layer -> out

        # embedding layer
        # inherit -> self.embedding_dict
        input_size = self._compute_interest_dim()
        # interest extractor layer
        self.interest_extractor = InterestExtractor(
            input_size=input_size, use_neg=use_negsampling, init_std=init_std
        )
        # interest evolution layer
        self.interest_evolution = InterestEvolving(
            input_size=input_size,
            gru_type=gru_type,
            use_neg=use_negsampling,
            init_std=init_std,
            att_hidden_size=att_hidden_units,
            att_activation=att_activation,
            att_weight_normalization=att_weight_normalization,
        )
        # DNN layer
        dnn_input_size = self._compute_dnn_dim() + input_size
        self.dnn = DNN(
            dnn_input_size,
            dnn_hidden_units,
            dnn_activation,
            l2_reg_dnn,
            dnn_dropout,
            use_bn,
            init_std=init_std,
            seed=seed,
        )
        self.linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        # prediction layer
        # inherit -> self.out

        # init
        for name, tensor in self.linear.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行兴趣抽取、兴趣演化与 DNN 部分的前向计算并输出预测结果。"""
        # [B, H] , [B, T, H], [B, T, H] , [B]
        query_emb, keys_emb, neg_keys_emb, keys_length = self._get_emb(X)
        # [b, T, H],  [1]  (b<H)
        masked_interest, aux_loss = self.interest_extractor(
            keys_emb, keys_length, neg_keys_emb
        )
        self.add_auxiliary_loss(aux_loss, self.alpha)
        # [B, H]
        hist = self.interest_evolution(query_emb, masked_interest, keys_length)
        # [B, H2]
        deep_input_emb = self._get_deep_input_emb(X)
        deep_input_emb = concat_fun([hist, deep_input_emb])
        dense_value_list = get_dense_input(
            X, self.feature_index, self.dense_feature_columns
        )
        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        # [B, 1]
        output = self.linear(self.dnn(dnn_input))
        y_pred = self.out(output)
        return y_pred

    def _get_emb(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """从输入中查找并拼接目标商品、历史行为及（可选）负采样序列的嵌入。"""
        # history feature columns : pos, neg
        history_feature_columns = []
        neg_history_feature_columns = []
        sparse_varlen_feature_columns = []
        history_fc_names = list(map(lambda x: "hist_" + x, self.item_features))
        neg_history_fc_names = list(map(lambda x: "neg_" + x, history_fc_names))
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in history_fc_names:
                history_feature_columns.append(fc)
            elif feature_name in neg_history_fc_names:
                neg_history_feature_columns.append(fc)
            else:
                sparse_varlen_feature_columns.append(fc)

        # convert input to emb
        features = self.feature_index
        query_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            features,
            self.sparse_feature_columns,
            return_feat_list=self.item_features,
            to_list=True,
        )
        # [batch_size, dim]
        query_emb = torch.squeeze(concat_fun(query_emb_list), 1)

        keys_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            features,
            history_feature_columns,
            return_feat_list=history_fc_names,
            to_list=True,
        )
        # [batch_size, max_len, dim]
        keys_emb = concat_fun(keys_emb_list)

        keys_length_feature_name = [
            feat.length_name
            for feat in self.varlen_sparse_feature_columns
            if feat.length_name is not None
        ]
        # [batch_size]
        keys_length = torch.squeeze(
            maxlen_lookup(X, features, keys_length_feature_name), 1
        )

        if self.use_negsampling:
            neg_keys_emb_list = embedding_lookup(
                X,
                self.embedding_dict,
                features,
                neg_history_feature_columns,
                return_feat_list=neg_history_fc_names,
                to_list=True,
            )
            neg_keys_emb = concat_fun(neg_keys_emb_list)
        else:
            neg_keys_emb = None

        return query_emb, keys_emb, neg_keys_emb, keys_length

    def _split_columns(self) -> None:
        """将 ``dnn_feature_columns`` 按类型拆分为稀疏、稠密与变长稀疏三类特征列。"""
        self.sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), self.dnn_feature_columns))
            if len(self.dnn_feature_columns)
            else []
        )
        self.dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), self.dnn_feature_columns))
            if len(self.dnn_feature_columns)
            else []
        )
        self.varlen_sparse_feature_columns = (
            list(
                filter(
                    lambda x: isinstance(x, VarLenSparseFeat), self.dnn_feature_columns
                )
            )
            if len(self.dnn_feature_columns)
            else []
        )

    def _compute_interest_dim(self) -> int:
        """计算历史行为特征拼接后的嵌入维度。"""
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.item_features:
                interest_dim += feat.embedding_dim
        return interest_dim

    def _compute_dnn_dim(self) -> int:
        """计算 DNN 输入的总维度（稀疏嵌入维度之和 + 稠密特征维度之和）。"""
        dnn_input_dim = 0
        for fc in self.sparse_feature_columns:
            dnn_input_dim += fc.embedding_dim
        for fc in self.dense_feature_columns:
            dnn_input_dim += fc.dimension
        return dnn_input_dim

    def _get_deep_input_emb(self, X: torch.Tensor) -> torch.Tensor:
        """查找除历史行为序列外的稀疏特征嵌入，用于 DNN 部分的输入拼接。"""
        dnn_input_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_feature_columns,
            mask_feat_list=self.item_features,
            to_list=True,
        )
        dnn_input_emb = concat_fun(dnn_input_emb_list)
        return dnn_input_emb.squeeze(1)


class InterestExtractor(nn.Module):
    """DIEN 的兴趣抽取层，使用 GRU 从历史行为序列中提取兴趣表示，
    并可选地通过辅助损失（auxiliary loss）利用负采样样本监督中间隐状态。
    """

    def __init__(
        self,
        input_size: int,
        use_neg: bool = False,
        init_std: float = 0.001,
        device: str = "cpu",
    ) -> None:
        super(InterestExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=input_size, batch_first=True
        )
        if self.use_neg:
            self.auxiliary_net = DNN(
                input_size * 2,
                [100, 50, 1],
                "sigmoid",
                init_std=init_std,
                device=device,
            )
        for name, tensor in self.gru.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)

    def forward(
        self,
        keys: torch.Tensor,
        keys_length: torch.Tensor,
        neg_keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        """对历史行为序列执行 GRU 编码，并在启用负采样时计算辅助损失。

        参数:
            keys: 形状为 ``[B, T, H]`` 的历史行为嵌入序列。
            keys_length: 形状为 ``[B]`` 的各样本有效序列长度。
            neg_keys: 形状为 ``[B, T, H]`` 的负采样序列嵌入，不使用负采样时为 ``None``。

        返回:
            masked_interests: 形状为 ``[b, H]`` 的兴趣表示（``b`` 为有效长度大于 0 的样本数）。
            aux_loss: 标量辅助损失。
        """
        batch_size, max_length, dim = keys.size()
        zero_outputs = torch.zeros(batch_size, dim, device=keys.device)
        aux_loss = torch.zeros((1,), device=keys.device)

        # create zero mask for keys_length, to make sure 'pack_padded_sequence' safe
        mask = keys_length > 0
        masked_keys_length = keys_length[mask]

        # batch_size validation check
        if masked_keys_length.shape[0] == 0:
            return (zero_outputs,)

        masked_keys = torch.masked_select(keys, mask.view(-1, 1, 1)).view(
            -1, max_length, dim
        )

        packed_keys = pack_padded_sequence(
            masked_keys,
            lengths=masked_keys_length.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(
            packed_interests,
            batch_first=True,
            padding_value=0.0,
            total_length=max_length,
        )

        if self.use_neg and neg_keys is not None:
            masked_neg_keys = torch.masked_select(neg_keys, mask.view(-1, 1, 1)).view(
                -1, max_length, dim
            )
            aux_loss = self._cal_auxiliary_loss(
                interests[:, :-1, :],
                masked_keys[:, 1:, :],
                masked_neg_keys[:, 1:, :],
                masked_keys_length - 1,
            )

        return interests, aux_loss

    def _cal_auxiliary_loss(
        self,
        states: torch.Tensor,
        click_seq: torch.Tensor,
        noclick_seq: torch.Tensor,
        keys_length: torch.Tensor,
    ) -> torch.Tensor:
        """基于正/负样本序列计算辅助二分类损失，用于监督 GRU 中间隐状态。"""
        # keys_length >= 1
        mask_shape = keys_length > 0
        keys_length = keys_length[mask_shape]
        if keys_length.shape[0] == 0:
            return torch.zeros((1,), device=states.device)

        _, max_seq_length, embedding_size = states.size()
        states = torch.masked_select(states, mask_shape.view(-1, 1, 1)).view(
            -1, max_seq_length, embedding_size
        )
        click_seq = torch.masked_select(click_seq, mask_shape.view(-1, 1, 1)).view(
            -1, max_seq_length, embedding_size
        )
        noclick_seq = torch.masked_select(noclick_seq, mask_shape.view(-1, 1, 1)).view(
            -1, max_seq_length, embedding_size
        )
        batch_size = states.size()[0]

        mask = (
            torch.arange(max_seq_length, device=states.device).repeat(batch_size, 1)
            < keys_length.view(-1, 1)
        ).float()

        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2

        click_p = (
            self.auxiliary_net(
                click_input.view(batch_size * max_seq_length, embedding_size)
            )
            .view(batch_size, max_seq_length)[mask > 0]
            .view(-1, 1)
        )
        click_target = torch.ones(
            click_p.size(), dtype=torch.float, device=click_p.device
        )

        noclick_p = (
            self.auxiliary_net(
                noclick_input.view(batch_size * max_seq_length, embedding_size)
            )
            .view(batch_size, max_seq_length)[mask > 0]
            .view(-1, 1)
        )
        noclick_target = torch.zeros(
            noclick_p.size(), dtype=torch.float, device=noclick_p.device
        )

        loss = F.binary_cross_entropy(
            torch.cat([click_p, noclick_p], dim=0),
            torch.cat([click_target, noclick_target], dim=0),
        )

        return loss


class InterestEvolving(nn.Module):
    """DIEN 的兴趣演化层，结合注意力机制与 GRU 变体（GRU/AIGRU/AGRU/AUGRU）
    对兴趣抽取层输出的序列做进一步演化建模。
    """

    __SUPPORTED_GRU_TYPE__ = ["GRU", "AIGRU", "AGRU", "AUGRU"]

    def __init__(
        self,
        input_size: int,
        gru_type: str = "GRU",
        use_neg: bool = False,
        init_std: float = 0.001,
        att_hidden_size: tuple[int, ...] = (64, 16),
        att_activation: str = "sigmoid",
        att_weight_normalization: bool = False,
    ) -> None:
        super(InterestEvolving, self).__init__()
        if gru_type not in InterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError("gru_type: {gru_type} is not supported")
        self.gru_type = gru_type
        self.use_neg = use_neg

        if gru_type == "GRU":
            self.attention = AttentionSequencePoolingLayer(
                embedding_dim=input_size,
                att_hidden_units=att_hidden_size,
                att_activation=att_activation,
                weight_normalization=att_weight_normalization,
                return_score=False,
            )
            self.interest_evolution = nn.GRU(
                input_size=input_size, hidden_size=input_size, batch_first=True
            )
        elif gru_type == "AIGRU":
            self.attention = AttentionSequencePoolingLayer(
                embedding_dim=input_size,
                att_hidden_units=att_hidden_size,
                att_activation=att_activation,
                weight_normalization=att_weight_normalization,
                return_score=True,
            )
            self.interest_evolution = nn.GRU(
                input_size=input_size, hidden_size=input_size, batch_first=True
            )
        elif gru_type == "AGRU" or gru_type == "AUGRU":
            self.attention = AttentionSequencePoolingLayer(
                embedding_dim=input_size,
                att_hidden_units=att_hidden_size,
                att_activation=att_activation,
                weight_normalization=att_weight_normalization,
                return_score=True,
            )
            self.interest_evolution = DynamicGRU(
                input_size=input_size, hidden_size=input_size, gru_type=gru_type
            )
        for name, tensor in self.interest_evolution.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    @staticmethod
    def _get_last_state(states: torch.Tensor, keys_length: torch.Tensor) -> torch.Tensor:
        """取出每个样本在其有效长度处的最后一个隐状态。"""
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()

        mask = torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1
        ) == (keys_length.view(-1, 1) - 1)

        return states[mask]

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        keys_length: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """根据所选 GRU 变体对兴趣序列做演化建模，输出融合后的兴趣表示。

        参数:
            query: 形状为 ``[B, H]`` 的目标商品嵌入。
            keys: 形状为 ``[b, T, H]`` 的兴趣抽取层输出序列（``masked_interests``）。
            keys_length: 形状为 ``[B]`` 的各样本有效序列长度。
            mask: 预留参数，当前未使用。

        返回:
            outputs: 形状为 ``[B, H]`` 的演化后兴趣表示。
        """
        batch_size, dim = query.size()
        max_length = keys.size()[1]

        # check batch validation
        zero_outputs = torch.zeros(batch_size, dim, device=query.device)
        mask = keys_length > 0
        # [B] -> [b]
        keys_length = keys_length[mask]
        if keys_length.shape[0] == 0:
            return zero_outputs

        # [B, H] -> [b, 1, H]
        query = torch.masked_select(query, mask.view(-1, 1)).view(-1, dim).unsqueeze(1)

        if self.gru_type == "GRU":
            packed_keys = pack_padded_sequence(
                keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(
                packed_interests,
                batch_first=True,
                padding_value=0.0,
                total_length=max_length,
            )
            outputs = self.attention(
                query, interests, keys_length.unsqueeze(1)
            )  # [b, 1, H]
            outputs = outputs.squeeze(1)  # [b, H]
        elif self.gru_type == "AIGRU":
            att_scores = self.attention(
                query, keys, keys_length.unsqueeze(1)
            )  # [b, 1, T]
            interests = keys * att_scores.transpose(1, 2)  # [b, T, H]
            packed_interests = pack_padded_sequence(
                interests,
                lengths=keys_length.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)  # [b, H]
        elif self.gru_type == "AGRU" or self.gru_type == "AUGRU":
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)).squeeze(
                1
            )  # [b, T]
            packed_interests = pack_padded_sequence(
                keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_scores = pack_padded_sequence(
                att_scores,
                lengths=keys_length.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True, padding_value=0.0, total_length=max_length
            )
            # pick last state
            outputs = InterestEvolving._get_last_state(outputs, keys_length)  # [b, H]
        # [b, H] -> [B, H]
        zero_outputs[mask] = outputs
        return zero_outputs
