import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import activation_layer
from .core import Conv2dSame
from .sequence import KMaxPooling


__all__ = [
    "FM",
    "BilinearInteraction",
    "SENETLayer",
    "CIN",
    "AFMLayer",
    "InteractingLayer",
    "CrossNet",
    "CrossNetMix",
    "InnerProductLayer",
    "OutterProductLayer",
    "ConvLayer",
    "LogTransformLayer",
    "BiInteractionPooling",
]


class FM(nn.Module):
    """因子分解机（FM），建模特征间不含线性项和偏置的二阶（成对）交互。

    输入形状:
        3D 张量，形状为 ``(batch_size, field_size, embedding_size)``。

    输出形状:
        2D 张量，形状为 ``(batch_size, 1)``。

    参考文献:
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self) -> None:
        super(FM, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算输入特征的二阶交互项。"""
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class BiInteractionPooling(nn.Module):
    """NFM 中使用的双线性交互层，将特征两两逐元素相乘后压缩为单个向量。

    输入形状:
        3D 张量，形状为 ``(batch_size, field_size, embedding_size)``。

    输出形状:
        3D 张量，形状为 ``(batch_size, 1, embedding_size)``。

    参考文献:
        - [He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](http://arxiv.org/abs/1708.05027)
    """

    def __init__(self) -> None:
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算特征两两逐元素乘积并压缩为单个向量。"""
        concated_embeds_value = inputs
        square_of_sum = torch.pow(
            torch.sum(concated_embeds_value, dim=1, keepdim=True), 2
        )
        sum_of_square = torch.sum(
            concated_embeds_value * concated_embeds_value, dim=1, keepdim=True
        )
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term


class SENETLayer(nn.Module):
    """FiBiNET 中使用的 SENET 层。

    输入形状:
        3D 张量，形状为 ``(batch_size, filed_size, embedding_size)``。

    输出形状:
        3D 张量，形状为 ``(batch_size, filed_size, embedding_size)``。

    参数:
        filed_size: 特征分组数量。
        reduction_ratio: 注意力网络输出空间的压缩比例。
        seed: 随机种子。
        device: 运行设备。

    参考文献:
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
    Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(
        self,
        filed_size: int,
        reduction_ratio: int = 3,
        seed: int = 1024,
        device: str = "cpu",
    ) -> None:
        super(SENETLayer, self).__init__()
        self.seed = seed
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU(),
        )
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算各特征场的 SENET 注意力权重并对输入做逐元素缩放。"""
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (len(inputs.shape))
            )
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))

        return V


class BilinearInteraction(nn.Module):
    """FiBiNET 中使用的双线性交互层。

    输入形状:
        3D 张量，形状为 ``(batch_size, filed_size, embedding_size)``。

    输出形状:
        3D 张量，形状为 ``(batch_size, filed_size*(filed_size-1)/2, embedding_size)``。

    参数:
        filed_size: 特征分组数量。
        embedding_size: 稀疏特征的嵌入维度。
        bilinear_type: 双线性函数的类型，可选 ``"all"``/``"each"``/``"interaction"``。
        seed: 随机种子。
        device: 运行设备。

    参考文献:
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
    Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(
        self,
        filed_size: int,
        embedding_size: int,
        bilinear_type: str = "interaction",
        seed: int = 1024,
        device: str = "cpu",
    ) -> None:
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.seed = seed
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(embedding_size, embedding_size, bias=False)
        elif self.bilinear_type == "each":
            for _ in range(filed_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False)
                )
        elif self.bilinear_type == "interaction":
            for _, _ in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False)
                )
        else:
            raise NotImplementedError
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """对各特征场两两组合执行双线性交互。"""
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (len(inputs.shape))
            )
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [
                torch.mul(self.bilinear(v_i), v_j)
                for v_i, v_j in itertools.combinations(inputs, 2)
            ]
        elif self.bilinear_type == "each":
            p = [
                torch.mul(self.bilinear[i](inputs[i]), inputs[j])
                for i, j in itertools.combinations(range(len(inputs)), 2)
            ]
        elif self.bilinear_type == "interaction":
            p = [
                torch.mul(bilinear(v[0]), v[1])
                for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)
            ]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)


class CIN(nn.Module):
    """xDeepFM 中使用的压缩交互网络（CIN）。

    输入形状:
        3D 张量，形状为 ``(batch_size, field_size, embedding_size)``。

    输出形状:
        2D 张量，形状为 ``(batch_size, featuremap_num)``；当 ``split_half=True`` 时
        ``featuremap_num = sum(layer_size[:-1]) // 2 + layer_size[-1]``，否则为 ``sum(layer_size)``。

    参数:
        field_size: 特征分组数量。
        layer_size: 各层特征图数量列表。
        activation: 作用于特征图的激活函数名称。
        split_half: 若为 ``False``，每个隐藏层中一半的特征图会连接到输出单元。
        l2_reg: L2 正则强度。
        seed: 随机种子。
        device: 运行设备。

    参考文献:
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(
        self,
        field_size: int,
        layer_size: tuple[int, ...] | list[int] = (128, 128),
        activation: str = "relu",
        split_half: bool = True,
        l2_reg: float = 1e-5,
        seed: int = 1024,
        device: str = "cpu",
    ) -> None:
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1"
            )

        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed

        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1)
            )

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True"
                    )

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        #         for tensor in self.conv1ds:
        #             nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """逐层计算压缩交互网络的特征图并拼接输出。"""
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (len(inputs.shape))
            )
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0
            x = torch.einsum("bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0])
            # x.shape = (batch_size , hi * m, dim)
            x = x.reshape(
                batch_size,
                hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1],
                dim,
            )
            # x.shape = (batch_size , hi, dim)
            x = self.conv1ds[i](x)

            if self.activation is None or self.activation == "linear":
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1
                    )
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)

        return result


class AFMLayer(nn.Module):
    """注意力因子分解机（AFM），建模特征间不含线性项和偏置的二阶（成对）交互。

    输入形状:
        3D 张量列表，每个张量形状为 ``(batch_size, 1, embedding_size)``。

    输出形状:
        2D 张量，形状为 ``(batch_size, 1)``。

    参数:
        in_features: 输入特征维度。
        attention_factor: 注意力网络输出空间的维度。
        l2_reg_w: 注意力网络的 L2 正则强度，取值范围 ``[0, 1)``。
        dropout_rate: 注意力网络输出的 dropout 比例，取值范围 ``[0, 1)``。
        seed: 随机种子。
        device: 运行设备。

    参考文献:
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(
        self,
        in_features: int,
        attention_factor: int = 4,
        l2_reg_w: float = 0,
        dropout_rate: float = 0,
        seed: int = 1024,
        device: str = "cpu",
    ) -> None:
        super(AFMLayer, self).__init__()
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        embedding_size = in_features

        self.attention_W = nn.Parameter(
            torch.Tensor(embedding_size, self.attention_factor)
        )

        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))

        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1))

        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))

        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(
                tensor,
            )

        for tensor in [self.attention_b]:
            nn.init.zeros_(
                tensor,
            )

        self.dropout = nn.Dropout(dropout_rate)

        self.to(device)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """基于注意力机制计算特征两两交互的加权和。"""
        embeds_vec_list = inputs
        row = []
        col = []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q

        bi_interaction = inner_product
        attention_temp = F.relu(
            torch.tensordot(bi_interaction, self.attention_W, dims=([-1], [0]))
            + self.attention_b
        )

        self.normalized_att_score = F.softmax(
            torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])), dim=1
        )
        attention_output = torch.sum(self.normalized_att_score * bi_interaction, dim=1)

        attention_output = self.dropout(attention_output)  # training

        afm_out = torch.tensordot(attention_output, self.projection_p, dims=([-1], [0]))
        return afm_out


class InteractingLayer(nn.Module):
    """AutoInt 中使用的层，通过多头自注意力机制建模不同特征场之间的相关性。

    输入形状:
        3D 张量，形状为 ``(batch_size, field_size, embedding_size)``。

    输出形状:
        3D 张量，形状为 ``(batch_size, field_size, embedding_size)``。

    参数:
        embedding_size: 输入特征维度。
        head_num: 多头自注意力网络的头数。
        use_res: 输出前是否使用标准残差连接。
        scaling: 是否对注意力得分进行缩放。
        seed: 随机种子。
        device: 运行设备。

    参考文献:
        - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(
        self,
        embedding_size: int,
        head_num: int = 2,
        use_res: bool = True,
        scaling: bool = False,
        seed: int = 1024,
        device: str = "cpu",
    ) -> None:
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError("head_num must be a int > 0")
        if embedding_size % head_num != 0:
            raise ValueError("embedding_size is not an integer multiple of head_num!")
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """基于多头自注意力计算特征场之间的交互表示。"""
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (len(inputs.shape))
            )

        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        # head_num None F F
        inner_product = torch.einsum("bnik,bnjk->bnij", querys, keys)
        if self.scaling:
            inner_product /= self.att_embedding_size**0.5

        # head_num None F F
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)
        # head_num None F D/head_num
        result = torch.matmul(self.normalized_att_scores, values)

        result = torch.cat(torch.split(result, 1), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)

        return result


class CrossNet(nn.Module):
    """Deep&Cross Network 中的 Cross 网络部分，可同时学习低阶和高阶交叉特征。

    输入形状:
        2D 张量，形状为 ``(batch_size, units)``。

    输出形状:
        2D 张量，形状为 ``(batch_size, units)``。

    参数:
        in_features: 输入特征维度。
        layer_num: 交叉层的层数。
        parameterization: 交叉网络的参数化方式，``"vector"`` 或 ``"matrix"``。
        seed: 随机种子。
        device: 运行设备。

    参考文献:
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(
        self,
        in_features: int,
        layer_num: int = 2,
        parameterization: str = "vector",
        seed: int = 1024,
        device: str = "cpu",
    ) -> None:
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == "vector":
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == "matrix":
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(
                torch.Tensor(self.layer_num, in_features, in_features)
            )
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """逐层计算显式特征交叉。"""
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == "vector":
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == "matrix":
                # W * xi  (bs, in_features, 1)
                xl_w = torch.matmul(self.kernels[i], x_l)

                # W * xi + b
                dot_ = xl_w + self.bias[i]

                # x0 · (W * xi + b) +xl  Hadamard-product
                x_l = x_0 * dot_ + x_l
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class CrossNetMix(nn.Module):
    """DCN-Mix 模型中的 Cross 网络部分，相较 DCN-M 的两点改进：
    1. 引入 MOE 以学习不同子空间中的特征交互；
    2. 在低维空间中加入非线性变换。

    输入形状:
        2D 张量，形状为 ``(batch_size, units)``。

    输出形状:
        2D 张量，形状为 ``(batch_size, units)``。

    参数:
        in_features: 输入特征维度。
        low_rank: 低秩空间的维度。
        num_experts: 专家数量。
        layer_num: 交叉层的层数。
        device: 运行设备，例如 ``"cpu"`` 或 ``"cuda:0"``。

    参考文献:
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(
        self,
        in_features: int,
        low_rank: int = 32,
        num_experts: int = 4,
        layer_num: int = 2,
        device: str = "cpu",
    ) -> None:
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = nn.Parameter(
            torch.Tensor(self.layer_num, num_experts, in_features, low_rank)
        )
        # V: (in_features, low_rank)
        self.V_list = nn.Parameter(
            torch.Tensor(self.layer_num, num_experts, in_features, low_rank)
        )
        # C: (low_rank, low_rank)
        self.C_list = nn.Parameter(
            torch.Tensor(self.layer_num, num_experts, low_rank, low_rank)
        )
        self.gating = nn.ModuleList(
            [nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)]
        )

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        init_para_list = [self.U_list, self.V_list, self.C_list]
        for para in init_para_list:
            for i in range(self.layer_num):
                nn.init.xavier_normal_(para[i])

        for i in range(len(self.bias)):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """基于混合低秩专家逐层计算特征交叉。"""
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(
                    self.V_list[i][expert_id].t(), x_l
                )  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(
                    self.U_list[i][expert_id], v_x
                )  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(
                output_of_experts, 2
            )  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(
                gating_score_of_experts, 1
            )  # (bs, num_experts, 1)
            moe_out = torch.matmul(
                output_of_experts, gating_score_of_experts.softmax(1)
            )
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


class InnerProductLayer(nn.Module):
    """PNN 中使用的内积层，计算特征向量两两之间的逐元素乘积或内积。

    输入形状:
        3D 张量列表，每个张量形状为 ``(batch_size, 1, embedding_size)``。

    输出形状:
        若 ``reduce_sum=True``，为 3D 张量，形状 ``(batch_size, N*(N-1)/2, 1)``；
        否则为 3D 张量，形状 ``(batch_size, N*(N-1)/2, embedding_size)``。

    参数:
        reduce_sum: 是否返回内积（``True``）而非逐元素乘积（``False``）。
        device: 运行设备。

    参考文献:
        - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//
        Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.]
        (https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self, reduce_sum: bool = True, device: str = "cpu") -> None:
        super(InnerProductLayer, self).__init__()
        self.reduce_sum = reduce_sum
        self.to(device)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """计算特征向量两两之间的内积或逐元素乘积。"""
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)

        inner_product = p * q
        if self.reduce_sum:
            inner_product = torch.sum(inner_product, dim=2, keepdim=True)
        return inner_product


class OutterProductLayer(nn.Module):
    """PNN 中使用的外积层，实现改编自论文作者发布在
    https://github.com/Atomu2014/product-nets 的代码。

    输入形状:
        N 个 3D 张量组成的列表，每个张量形状为 ``(batch_size, 1, embedding_size)``。

    输出形状:
        2D 张量，形状为 ``(batch_size, N*(N-1)/2)``。

    参数:
        field_size: 特征分组数量。
        embedding_size: 稀疏特征的嵌入维度。
        kernel_type: 核权重矩阵类型，可选 ``"mat"``/``"vec"``/``"num"``。
        seed: 随机种子。
        device: 运行设备。

    参考文献:
        - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(
        self,
        field_size: int,
        embedding_size: int,
        kernel_type: str = "mat",
        seed: int = 1024,
        device: str = "cpu",
    ) -> None:
        super(OutterProductLayer, self).__init__()
        self.kernel_type = kernel_type

        num_inputs = field_size
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        embed_size = embedding_size
        if self.kernel_type == "mat":
            self.kernel = nn.Parameter(torch.Tensor(embed_size, num_pairs, embed_size))

        elif self.kernel_type == "vec":
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, embed_size))

        elif self.kernel_type == "num":
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, 1))
        nn.init.xavier_uniform_(self.kernel)

        self.to(device)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """计算特征向量两两之间的外积交互。"""
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)

        # -------------------------
        if self.kernel_type == "mat":
            p.unsqueeze_(dim=1)
            # k     k* pair* k
            # batch * pair
            kp = torch.sum(
                # batch * pair * k
                torch.mul(
                    # batch * pair * k
                    torch.transpose(
                        # batch * k * pair
                        torch.sum(
                            # batch * k * pair * k
                            torch.mul(p, self.kernel),
                            dim=-1,
                        ),
                        2,
                        1,
                    ),
                    q,
                ),
                dim=-1,
            )
        else:
            # 1 * pair * (k or 1)
            k = torch.unsqueeze(self.kernel, 0)

            # batch * pair
            kp = torch.sum(p * q * k, dim=-1)

            # p q # b * p * k

        return kp


class ConvLayer(nn.Module):
    """CCPM 中使用的卷积层。

    输入形状:
        N 个 3D 张量组成的列表，每个张量形状为 ``(batch_size, 1, filed_size, embedding_size)``。

    输出形状:
        N 个 3D 张量组成的列表，每个张量形状为 ``(batch_size, last_filters, pooling_size, embedding_size)``。

    参数:
        field_size: 特征分组数量。
        conv_kernel_width: 各卷积层滤波器宽度组成的列表，可为空列表。
        conv_filters: 各卷积层滤波器数量组成的列表，可为空列表。
        device: 运行设备。

    参考文献:
        - Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.(http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)
    """

    def __init__(
        self,
        field_size: int,
        conv_kernel_width: list[int],
        conv_filters: list[int],
        device: str = "cpu",
    ) -> None:
        super(ConvLayer, self).__init__()
        self.device = device
        module_list = []
        n = int(field_size)
        l = len(conv_filters)  # noqa: E741
        filed_shape = n
        for i in range(1, l + 1):
            if i == 1:
                in_channels = 1
            else:
                in_channels = conv_filters[i - 2]
            out_channels = conv_filters[i - 1]
            width = conv_kernel_width[i - 1]
            k = max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3
            module_list.append(
                Conv2dSame(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(width, 1),
                    stride=1,
                ).to(self.device)
            )
            module_list.append(torch.nn.Tanh().to(self.device))

            # KMaxPooling, extract top_k, returns tensors values
            module_list.append(
                KMaxPooling(k=min(k, filed_shape), axis=2, device=self.device).to(
                    self.device
                )
            )
            filed_shape = min(k, filed_shape)
        self.conv_layer = nn.Sequential(*module_list)
        self.to(device)
        self.filed_shape = filed_shape

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """依次执行各卷积层，输出池化后的特征图。"""
        return self.conv_layer(inputs)


class LogTransformLayer(nn.Module):
    """自适应因子分解网络（AFN）中的对数变换层，用于建模任意阶的交叉特征。

    输入形状:
        3D 张量，形状为 ``(batch_size, field_size, embedding_size)``。

    输出形状:
        2D 张量，形状为 ``(batch_size, ltl_hidden_size*embedding_size)``。

    参数:
        field_size: 特征分组数量。
        embedding_size: 稀疏特征的嵌入维度。
        ltl_hidden_size: AFN 中对数神经元的数量。

    参考文献:
        - Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
        Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
    """

    def __init__(self, field_size: int, embedding_size: int, ltl_hidden_size: int) -> None:
        super(LogTransformLayer, self).__init__()

        self.ltl_weights = nn.Parameter(torch.Tensor(field_size, ltl_hidden_size))
        self.ltl_biases = nn.Parameter(torch.Tensor(1, 1, ltl_hidden_size))
        self.bn = nn.ModuleList([nn.BatchNorm1d(embedding_size) for i in range(2)])
        nn.init.normal_(self.ltl_weights, mean=0.0, std=0.1)
        nn.init.zeros_(self.ltl_biases)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """对输入特征执行对数-指数变换以建模任意阶交叉特征。"""
        # Avoid numeric overflow
        afn_input = torch.clamp(torch.abs(inputs), min=1e-7, max=float("Inf"))
        # Transpose to shape: ``(batch_size,embedding_size,field_size)``
        afn_input_trans = torch.transpose(afn_input, 1, 2)
        # Logarithmic transformation layer
        ltl_result = torch.log(afn_input_trans)
        ltl_result = self.bn[0](ltl_result)
        ltl_result = torch.matmul(ltl_result, self.ltl_weights) + self.ltl_biases
        ltl_result = torch.exp(ltl_result)
        ltl_result = self.bn[1](ltl_result)
        ltl_result = torch.flatten(ltl_result, start_dim=1)
        return ltl_result
