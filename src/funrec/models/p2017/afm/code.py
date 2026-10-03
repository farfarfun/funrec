# -*- coding:utf-8 -*-
"""
Reference:
    [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.
    (https://arxiv.org/abs/1708.04617)
"""

from typing import Any

import torch

from funrec.layers import FM, AFMLayer
from funrec.models.b2000 import BaseModel


class AFM(BaseModel):
    """注意力因子分解机（AFM）架构。

    参数:
        linear_feature_columns: 线性部分使用的特征列。
        dnn_feature_columns: 深度部分使用的特征列。
        use_attention: 是否使用注意力机制；为 ``False`` 时退化为标准 FM。
        attention_factor: 注意力网络的单元数，正整数。
        l2_reg_linear: 线性部分的 L2 正则强度。
        l2_reg_embedding: 嵌入向量的 L2 正则强度。
        l2_reg_att: 注意力网络的 L2 正则强度。
        afm_dropout: 注意力网络输出的 dropout 比例，取值范围 ``[0, 1)``。
        init_std: 嵌入向量初始化的标准差。
        seed: 随机种子。
        task: 任务类型，``"binary"`` 对应二分类 logloss，``"regression"`` 对应回归损失。
        device: 运行设备，``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练使用的 GPU 列表；为 ``None`` 时仅使用 ``device``，
            否则 ``gpus[0]`` 需与 ``device`` 一致。

    参考文献:
        [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.
        (https://arxiv.org/abs/1708.04617)
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        use_attention: bool = True,
        attention_factor: int = 8,
        l2_reg_linear: float = 1e-5,
        l2_reg_embedding: float = 1e-5,
        l2_reg_att: float = 1e-5,
        afm_dropout: float = 0,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
    ) -> None:
        super(AFM, self).__init__(
            linear_feature_columns,
            dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
        )

        self.use_attention = use_attention

        if use_attention:
            self.fm = AFMLayer(
                self.embedding_size,
                attention_factor,
                l2_reg_att,
                afm_dropout,
                seed,
                device,
            )
            self.add_regularization_weight(self.fm.attention_W, l2=l2_reg_att)
        else:
            self.fm = FM()

        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """执行线性部分与（注意力）FM 交互部分的前向计算并融合输出。"""
        sparse_embedding_list, _ = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict, support_dense=False
        )
        logit = self.linear_model(X)
        if len(sparse_embedding_list) > 0:
            if self.use_attention:
                logit += self.fm(sparse_embedding_list)
            else:
                logit += self.fm(torch.cat(sparse_embedding_list, dim=1))

        y_pred = self.out(logit)

        return y_pred
