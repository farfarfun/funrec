from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from funrec.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from funrec.models import DeepFM

DATA_DIR = Path(__file__).resolve().parent


def pad_sequences(
    sequences: list[list[int]], maxlen: int, padding: str = "post"
) -> np.ndarray:
    """将变长的整数序列填充为定长的二维数组（右侧补 0）。

    避免引入整个 keras/tensorflow 依赖，仅实现本示例需要的最小填充逻辑。

    参数:
        sequences: 待填充的整数序列列表。
        maxlen: 填充后的统一长度。
        padding: 填充方向，``"post"`` 在序列末尾补 0，``"pre"`` 在序列开头补 0。

    返回:
        形状为 ``(len(sequences), maxlen)`` 的 int32 数组。
    """
    result = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        trimmed = seq[-maxlen:]
        if padding == "post":
            result[i, : len(trimmed)] = trimmed
        else:
            result[i, maxlen - len(trimmed) :] = trimmed
    return result


def split(x):
    key_ans = x.split("|")
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
    data = pd.read_csv(DATA_DIR / "movielens_sample.txt")
    sparse_features = [
        "movie_id",
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip",
    ]
    target = ["rating"]

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data["genres"].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(
        genres_list,
        maxlen=max_len,
        padding="post",
    )

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [
        SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
        for feat in sparse_features
    ]

    varlen_feature_columns = [
        VarLenSparseFeat(
            SparseFeat("genres", vocabulary_size=len(key2index) + 1, embedding_dim=4),
            maxlen=max_len,
            combiner="mean",
        )
    ]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in sparse_features}  #
    model_input["genres"] = genres_list

    # 4.Define Model,compile and train

    device = "cpu"
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print("cuda ready...")
        device = "cuda:0"

    model = DeepFM(
        linear_feature_columns, dnn_feature_columns, task="regression", device=device
    )

    model.compile(
        "adam",
        "mse",
        metrics=["mse"],
    )
    history = model.fit(
        model_input,
        data[target].values,
        batch_size=256,
        epochs=10,
        verbose=2,
        validation_split=0.2,
    )
