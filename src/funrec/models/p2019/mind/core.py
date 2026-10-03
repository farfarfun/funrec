# !/usr/bin/python
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class CoreCapsuleNetwork(nn.Module):
    """多兴趣召回模型共用的胶囊网络基类，实现动态路由（dynamic routing）的核心逻辑。

    子类（如 ``MINDCapsuleNetwork``、``BCapsuleNetwork``、``ComiCapsuleNetwork``）
    只需实现各自的候选胶囊计算方式（``item_eb_hat``/``capsule_weight`` 的构造），
    路由迭代过程复用本类的 ``forward``。

    参数:
        hidden_size: 兴趣向量的隐藏维度。
        seq_len: 用户历史行为序列的最大长度。
        interest_num: 提取的兴趣胶囊数量。
        routing_times: 动态路由迭代次数。
        hard_readout: 是否在读出阶段使用 hard attention（当前基类未直接使用，供子类参考）。
        relu_layer: 是否在输出兴趣向量前额外接一层 ReLU 线性变换。
    """

    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        interest_num: int = 4,
        routing_times: int = 3,
        hard_readout: bool = True,
        relu_layer: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(CoreCapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size  # h
        self.seq_len = seq_len  # s
        self.interest_num = interest_num  # i
        self.routing_times = routing_times
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU()
        )

    def forward(
        self,
        mask: torch.Tensor,
        item_eb_hat: torch.Tensor,
        capsule_weight: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """对候选胶囊执行多轮动态路由，输出最终的兴趣胶囊表示。

        参数:
            mask: 形状为 ``[batch_size, seq_len]`` 的行为序列掩码。
            item_eb_hat: 候选胶囊输入，形状可 reshape 为
                ``[batch_size, seq_len, interest_num, hidden_size]``。
            capsule_weight: 路由 logits，形状为 ``[batch_size, interest_num, seq_len]``。

        返回:
            形状为 ``[batch_size, interest_num, hidden_size]`` 的兴趣胶囊。
        """
        item_eb_hat = torch.reshape(
            item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size)
        )
        item_eb_hat = torch.transpose(item_eb_hat, 2, 1)

        # shape=[b, i, s, h]
        if self.stop_grad:  # 截断反向传播，item_emb_hat不计入梯度计算中
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        # 动态路由传播3次
        for i in range(self.routing_times):
            # [b, i, s]
            attention_mask = torch.repeat_interleave(
                torch.unsqueeze(mask, 1), self.interest_num, 1
            )
            paddings = torch.zeros_like(attention_mask)

            # 计算c，进行mask，最后shape=[b, i, 1, s]
            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(
                attention_mask == 0, paddings, capsule_softmax_weight
            )

            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < self.routing_times - 1:
                # s=c*u_hat , (b, i, 1, s) * (b, i, s, h)
                # shape=(b, i, 1, h)
                interest_capsule = torch.matmul(
                    capsule_softmax_weight, item_eb_hat_iter
                )

                # shape=(b, i, 1, 1)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, keepdim=True)

                # shape=(b, i, 1, 1)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)

                # squash(s)->v,shape=(b, i, 1, h)
                interest_capsule = scalar_factor * interest_capsule

                # 更新b

                # u_hat*v, shape=(b, i, s, 1)
                delta_weight = torch.matmul(
                    item_eb_hat_iter,
                    torch.transpose(interest_capsule, 3, 2),
                    # shape=(b, i, h, 1)
                )

                # shape=(b, i, s)
                delta_weight = torch.reshape(
                    delta_weight, (-1, self.interest_num, self.seq_len)
                )
                capsule_weight = capsule_weight + delta_weight  # 更新b
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(
            interest_capsule, (-1, self.interest_num, self.hidden_size)
        )

        # MIND模型使用book数据库时，使用relu_layer
        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)
        return interest_capsule


class MINDCapsuleNetwork(CoreCapsuleNetwork):
    """MIND 模型使用的胶囊网络，通过共享的线性变换 + 重复广播构造候选胶囊。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MINDCapsuleNetwork, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self, item_eb: torch.Tensor, mask: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """基于共享线性变换构造候选胶囊并执行动态路由。"""
        # [b, s, h]
        item_eb_hat = self.linear(item_eb)
        # [b, s, h*in]
        item_eb_hat = torch.repeat_interleave(item_eb_hat, self.interest_num, 2)
        capsule_weight = torch.randn(
            (item_eb_hat.shape[0], self.interest_num, self.seq_len)
        )
        return super().forward(mask, item_eb_hat, capsule_weight)


class BCapsuleNetwork(CoreCapsuleNetwork):
    """B2I（behavior-to-interest）胶囊网络，使用单层线性变换一次性生成全部候选胶囊。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(
            self.hidden_size, self.hidden_size * self.interest_num, bias=False
        )

    def forward(
        self, item_eb: torch.Tensor, mask: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """构造候选胶囊并执行动态路由。"""
        item_eb_hat = self.linear(item_eb)
        capsule_weight = torch.zeros(
            (item_eb_hat.shape[0], self.interest_num, self.seq_len)
        )
        return super().forward(mask, item_eb_hat, capsule_weight)


class MIND(nn.Module):
    """MIND 多兴趣召回模型，通过胶囊网络从用户历史行为序列中提取多个兴趣向量，
    并结合标签感知注意力（label-aware attention）选择与目标物品最相关的兴趣做训练。

    参数:
        embedding_dim: 物品嵌入维度。
        max_length: 用户历史行为序列的最大长度。
        n_items: 物品词表大小。
        interest_num: 提取的兴趣胶囊数量。

    参考文献:
        [1] Li C, Liu Z, Wu M, et al. Multi-interest network with dynamic routing for recommendation at Tmall[C]//
             Proceedings of the 28th ACM international conference on information and knowledge management. 2019: 2615-2623.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        n_items: int,
        interest_num: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(MIND, self).__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.n_items = n_items

        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        # capsule network
        self.capsule = MINDCapsuleNetwork(
            self.embedding_dim,
            self.max_length,
            interest_num=interest_num,
        )
        self.loss_fun = nn.CrossEntropyLoss()
        self.reset_parameters()

    def calculate_loss(
        self, user_emb: torch.Tensor, pos_item: torch.Tensor
    ) -> torch.Tensor:
        """计算最优兴趣向量与正样本物品之间的交叉熵损失。"""
        all_items = self.item_emb.weight
        scores = torch.matmul(user_emb, all_items.transpose(1, 0))
        pos_item = pos_item.squeeze(1).long()
        return self.loss_fun(scores, pos_item)

    def output_items(self) -> torch.Tensor:
        """返回物品嵌入矩阵，供召回阶段计算相似度使用。"""
        return self.item_emb.weight

    def reset_parameters(self, initializer: Any = None) -> None:
        """使用 Kaiming 正态分布重新初始化模型的全部可学习参数。"""
        for weight in self.parameters():
            if len(weight.shape) < 2:
                torch.nn.init.kaiming_normal_(weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_normal_(weight)

    def forward(
        self,
        item_seq: torch.Tensor,
        mask: torch.Tensor,
        item: torch.Tensor,
        train: bool = True,
    ) -> dict[str, torch.Tensor]:
        """执行多兴趣提取与标签感知注意力，训练模式下同时返回损失。"""
        if train:
            # 1. embedding layer
            item_seq = item_seq.long()
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            item_e = self.item_emb(item.long()).squeeze(1)

            # 2. multi-interest extractor layer + 3. label-aware attention layer
            multi_interest_emb = self.capsule(seq_emb, mask)  # Batch,K,Emb
            cos_res = torch.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            # 取内积结果最大的，作为最后的得分，并且取出对应的index
            k_index = torch.argmax(cos_res, dim=1)
            best_interest_emb = torch.rand(
                (multi_interest_emb.shape[0], multi_interest_emb.shape[2])
            )
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]

            # 4. loss function
            loss = self.calculate_loss(best_interest_emb, item)
            output_dict = {
                "user_emb": multi_interest_emb,
                "loss": loss,
            }
        else:
            # test stage
            item_seq = item_seq.long()
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            multi_interest_emb = self.capsule(seq_emb, mask)  # Batch,K,Emb
            output_dict = {
                "user_emb": multi_interest_emb,
            }
        return output_dict
