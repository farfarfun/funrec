import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
from farlog import getLogger
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .core import MIND

logger = getLogger("funrec")


class SeqnenceDataset(Dataset):
    """为 MIND 训练或评估构造用户行为序列数据集。

    参数:
        config: 包含 ``max_length`` 的训练配置。
        df: 含有 ``user_id``、``item_id`` 和 ``timestamp`` 列的行为数据。
        phase: ``"train"`` 时随机采样训练目标，``"test"`` 时保留后 20% 物品。
    """

    def __init__(
        self, config: Mapping[str, Any], df: Any, phase: Literal["train", "test"] = "train"
    ) -> None:
        self.config = config
        self.df = df
        self.max_length = self.config["max_length"]
        self.df = self.df.sort_values(by=["user_id", "timestamp"])
        self.user2item = self.df.groupby("user_id")["item_id"].apply(list).to_dict()
        self.user_list = self.df["user_id"].unique()
        self.phase = phase

    def __len__(self) -> int:
        return len(self.user2item)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | list[Any]]:
        if self.phase == "train":
            user_id = self.user_list[index]
            item_list = self.user2item[user_id]
            hist_item_list = []
            hist_mask_list = []

            k = random.choice(
                range(4, len(item_list))
            )  # 从[8,len(item_list))中随机选择一个index
            # k = np.random.randint(2,len(item_list))
            item_id = item_list[k]  # 该index对应的item加入item_id_list

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length : k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))

            return (
                torch.Tensor(hist_item_list).squeeze(0),
                torch.Tensor(hist_mask_list).squeeze(0),
                torch.Tensor([item_id]),
            )
        else:
            user_id = self.user_list[index]
            item_list = self.user2item[user_id]
            hist_item_list = []
            hist_mask_list = []

            k = int(0.8 * len(item_list))
            # k = len(item_list)-1

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length : k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))

            return (
                torch.Tensor(hist_item_list).squeeze(0),
                torch.Tensor(hist_mask_list).squeeze(0),
                item_list[k:],
            )

    def get_test_gd(self) -> dict[Any, list[Any]]:
        """获取每个用户留作评估的后 20% 交互物品。

        返回:
            用户 ID 到其评估物品列表的映射。
        """
        self.test_gd = {}
        for user in self.user2item:
            item_list = self.user2item[user]
            test_item_index = int(0.8 * len(item_list))
            self.test_gd[user] = item_list[test_item_index:]
        return self.test_gd


config = {
    "train_path": "./data/data173799/train_enc.csv",
    "valid_path": "./data/data173799/valid_enc.csv",
    "test_path": "./data/data173799/test_enc.csv",
    "lr": 1e-4,
    "Epoch": 5,
    "batch_size": 256,
    "embedding_dim": 16,
    "num_layers": 1,
    "max_length": 20,
    "n_items": 15406,
    "K": 4,
}


def my_collate(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, list[Any]]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[list[Any], ...]]:
    """将评估样本合并为张量批次，同时保留每个用户的目标物品列表。

    参数:
        batch: 数据集返回的历史物品、掩码和目标物品列表组成的样本序列。

    返回:
        历史物品张量、历史掩码张量及按用户分组的目标物品元组。
    """
    hist_item, hist_mask, item_list = list(zip(*batch))

    hist_item = [x.unsqueeze(0) for x in hist_item]
    hist_mask = [x.unsqueeze(0) for x in hist_mask]

    hist_item = torch.cat(hist_item, axis=0)
    hist_mask = torch.cat(hist_mask, axis=0)
    return hist_item, hist_mask, item_list


def save_model(model: MIND, path: str) -> None:
    """将 MIND 模型参数保存到指定目录。

    参数:
        model: 待保存的 MIND 模型。
        path: 模型目录，目录不存在时会自动创建。

    返回:
        无返回值。
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path + "model.pth")


def load_model(model: MIND, path: str) -> MIND:
    """从指定目录加载 MIND 模型参数。

    参数:
        model: 接收已保存参数的 MIND 模型实例。
        path: 包含 ``model.pth`` 的模型目录。

    返回:
        已加载参数的模型实例。
    """
    state_dict = torch.load(path + "model.pth")
    model.load_state_dict(state_dict)
    logger.info("模型已从 {} 加载", path)
    return model


"""
note: 基于faiss的向量召回
"""


def get_predict(
    model: MIND,
    test_data: Iterable[tuple[torch.Tensor, torch.Tensor, Sequence[Sequence[int]]]],
    hidden_size: int,
    topN: int = 20,
) -> tuple[dict[int, Sequence[int]], dict[int, Sequence[int]]]:
    """使用 FAISS 为评估用户检索 Top-N 推荐物品。

    参数:
        model: 用于生成用户和物品嵌入的 MIND 模型。
        test_data: 由历史物品、掩码和真实目标物品构成的评估批次。
        hidden_size: 物品嵌入维度。
        topN: 每位用户最多返回的推荐物品数。

    返回:
        真实目标物品映射和对应的推荐物品映射。
    """
    import faiss

    item_embs = model.output_items().cpu().detach().numpy()
    item_embs = normalize(item_embs, norm="l2")
    gpu_index = faiss.IndexFlatIP(hidden_size)
    gpu_index.add(item_embs)

    test_gd = {}
    preds = {}

    user_id = 0

    for item_seq, mask, targets in tqdm(test_data):
        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        user_embs = model(item_seq, mask, None, train=False)["user_emb"]
        user_embs = user_embs.cpu().detach().numpy()

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2:  # 非多兴趣模型评估
            user_embs = normalize(user_embs, norm="l2").astype("float32")
            D, I = gpu_index.search(
                user_embs, topN
            )  # Inner Product近邻搜索，D为distance，I是index
            #             D,I = faiss.knn(user_embs, item_embs, topN,metric=faiss.METRIC_INNER_PRODUCT)
            for i, iid_list in enumerate(
                targets
            ):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                test_gd[user_id] = iid_list
                preds[user_id] = I[i, :]
                user_id += 1
        else:  # 多兴趣模型评估
            ni = user_embs.shape[1]  # num_interest
            user_embs = np.reshape(
                user_embs, [-1, user_embs.shape[-1]]
            )  # shape=(batch_size*num_interest, embedding_dim)
            user_embs = normalize(user_embs, norm="l2").astype("float32")
            # Inner Product近邻搜索，D为distance，I是index
            D, I = gpu_index.search(user_embs, topN)
            #             D,I = faiss.knn(user_embs, item_embs, topN,metric=faiss.METRIC_INNER_PRODUCT)

            # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
            for i, iid_list in enumerate(targets):
                item_list_set = []

                # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                item_list = list(
                    zip(
                        np.reshape(I[i * ni : (i + 1) * ni], -1),
                        np.reshape(D[i * ni : (i + 1) * ni], -1),
                    )
                )
                # 降序排序，内积越大，向量越近
                item_list.sort(key=lambda x: x[1], reverse=True)
                # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                test_gd[user_id] = iid_list
                preds[user_id] = item_list_set
                user_id += 1
    return test_gd, preds


def evaluate(
    preds: Mapping[int, Sequence[int]], test_gd: Mapping[int, Sequence[int]], topN: int = 50
) -> dict[str, float]:
    """计算召回结果的 Recall、NDCG 和 HitRate 指标。

    参数:
        preds: 用户 ID 到预测推荐物品序列的映射。
        test_gd: 用户 ID 到真实目标物品序列的映射。
        topN: 参与指标计算的推荐结果数。

    返回:
        包含 ``recall``、``ndcg`` 和 ``hitrate`` 的指标字典。
    """
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for user in test_gd:
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:topN]:
                recall += 1
                dcg += 1.0 / math.log2(no + 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log2(no + 2)
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    return {f"recall@{topN}": recall, f"ndcg@{topN}": ndcg, f"hitrate@{topN}": hitrate}


# 指标计算
def evaluate_model(
    model: MIND,
    test_loader: Iterable[tuple[torch.Tensor, torch.Tensor, Sequence[Sequence[int]]]],
    embedding_dim: int,
    topN: int = 20,
) -> dict[str, float]:
    """对 MIND 模型执行向量召回并计算评估指标。

    参数:
        model: 待评估的 MIND 模型。
        test_loader: 由评估数据集生成的批次迭代器。
        embedding_dim: 物品嵌入维度。
        topN: 每位用户参与评估的推荐结果数。

    返回:
        包含 Recall、NDCG 和 HitRate 的指标字典。
    """
    test_gd, preds = get_predict(model, test_loader, embedding_dim, topN=topN)
    return evaluate(preds, test_gd, topN=topN)


def plot_embedding(data: np.ndarray, title: str) -> None:
    """绘制二维物品嵌入散点图。

    参数:
        data: 形状为 ``[n_samples, 2]`` 的二维嵌入数组。
        title: 图表标题。

    返回:
        无返回值。
    """
    from matplotlib import pyplot as plt

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    plt.figure(dpi=120)
    plt.scatter(data[:, 0], data[:, 1], marker=".")

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


def main(training_config: dict[str, object] | None = None) -> None:
    """读取配置的数据，训练 MIND 模型并输出评估与嵌入图。

    参数:
        training_config: 可选的训练配置；未提供时使用模块默认配置。

    返回:
        无返回值。
    """
    import pandas as pd
    from sklearn.manifold import TSNE

    active_config = config if training_config is None else training_config
    train_df = pd.read_csv(active_config["train_path"])
    valid_df = pd.read_csv(active_config["valid_path"])
    test_df = pd.read_csv(active_config["test_path"])
    train_dataset = SeqnenceDataset(active_config, train_df, phase="train")
    valid_dataset = SeqnenceDataset(active_config, valid_df, phase="test")
    test_dataset = SeqnenceDataset(active_config, test_df, phase="test")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=active_config["batch_size"],
        shuffle=True,
        num_workers=8,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=active_config["batch_size"],
        shuffle=False,
        collate_fn=my_collate,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=active_config["batch_size"],
        shuffle=False,
        collate_fn=my_collate,
    )

    model = MIND(
        embedding_dim=active_config["embedding_dim"],
        max_length=active_config["max_length"],
        n_items=active_config["n_items"],
        interest_num=active_config.get("K", 5),
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=active_config["lr"])
    log_df = pd.DataFrame()
    best_reacall = -1
    exp_path = "./ml-20m_softmax/MIND_{}_{}_{}/".format(
        active_config["lr"],
        active_config["batch_size"],
        active_config["embedding_dim"],
    )
    os.makedirs(exp_path, exist_ok=True, mode=0o777)
    patience = 5
    last_improve_epoch = 1
    log_csv = exp_path + "log.csv"

    for epoch in range(1, 1 + active_config["Epoch"]):
        pbar = tqdm(train_loader)
        model.train()
        loss_list = []
        logger.info("开始训练")
        for item_seq, mask, item in pbar:
            loss = model(item_seq, mask, item)["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
            pbar.set_description("Epoch [{}/{}]".format(epoch, active_config["Epoch"]))
            pbar.set_postfix(loss=np.mean(loss_list))

        logger.info("验证中")
        recall_metric = evaluate_model(
            model, valid_loader, active_config["embedding_dim"], topN=50
        )
        logger.info("验证指标：{}", recall_metric)
        recall_metric["phase"] = "valid"
        log_df = pd.concat([log_df, pd.DataFrame([recall_metric])], ignore_index=True)
        log_df.to_csv(log_csv)

        if recall_metric["recall@50"] > best_reacall:
            save_model(model, exp_path)
            best_reacall = recall_metric["recall@50"]
            last_improve_epoch = epoch
        if epoch - last_improve_epoch > patience:
            break

    logger.info("测试中")
    model = load_model(model, exp_path)
    recall_metric = evaluate_model(
        model, test_loader, active_config["embedding_dim"], topN=50
    )
    logger.info("测试指标：{}", recall_metric)
    recall_metric["phase"] = "test"
    log_df = pd.concat([log_df, pd.DataFrame([recall_metric])], ignore_index=True)
    log_df.to_csv(log_csv)

    item_emb = model.output_items().detach().numpy()
    tsne_emb = TSNE(n_components=2).fit_transform(item_emb)
    plot_embedding(tsne_emb, "MIND Item Embedding")


if __name__ == "__main__":
    main()
