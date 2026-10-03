# -*- coding:utf-8 -*-


import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from farlog import getLogger
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from funrec.callbacks import CallbackList, History
from funrec.inputs import (
    DenseFeat,
    SparseFeat,
    VarLenSparseFeat,
    build_input_features,
    create_embedding_matrix,
    get_varlen_pooling_list,
    varlen_embedding_lookup,
)
from funrec.layers import PredictionLayer
from funrec.layers.utils import slice_arrays

from .line import Linear

logger = getLogger("funrec")


class BaseModel(nn.Module):
    """funrec 中所有 CTR/推荐模型的基类，封装了特征嵌入、线性部分、训练/评估/
    预测循环、正则化、回调等公共逻辑，具体模型只需在子类中组合 ``embedding_dict``/
    ``linear_model`` 与自定义的深度网络结构。

    参数:
        linear_feature_columns: 线性部分使用的特征列（``SparseFeat``/``DenseFeat``/
            ``VarLenSparseFeat``）。
        dnn_feature_columns: 深度网络部分使用的特征列。
        l2_reg_linear: 线性部分权重的 L2 正则强度。
        l2_reg_embedding: 嵌入层权重的 L2 正则强度。
        init_std: 嵌入层权重正态初始化的标准差。
        seed: 随机种子。
        task: 任务类型，``"binary"``/``"multiclass"``/``"regression"``。
        device: 运行设备，例如 ``"cpu"`` 或 ``"cuda:0"``。
        gpus: 多卡训练时使用的 GPU 设备列表，第一个元素需与 ``device`` 一致。
    """

    def __init__(
        self,
        linear_feature_columns: list[Any],
        dnn_feature_columns: list[Any],
        l2_reg_linear: float = 1e-5,
        l2_reg_embedding: float = 1e-5,
        init_std: float = 0.0001,
        seed: int = 1024,
        task: str = "binary",
        device: str = "cpu",
        gpus: list[int | torch.device] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(BaseModel, self).__init__(*args, **kwargs)
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns
        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError("`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns
        )
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(
            dnn_feature_columns, init_std, sparse=False, device=device
        )
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device
        )

        self.regularization_weight = []

        self.add_regularization_weight(
            self.embedding_dict.parameters(), l2=l2_reg_embedding
        )
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)
        self.out = PredictionLayer(task)
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

    def fit(
        self,
        x: list[np.ndarray] | dict[str, np.ndarray] | None = None,
        y: np.ndarray | None = None,
        batch_size: int | None = None,
        epochs: int = 1,
        verbose: int = 1,
        initial_epoch: int = 0,
        validation_split: float = 0.0,
        validation_data: tuple[Any, ...] | None = None,
        shuffle: bool = True,
        callbacks: list[Any] | None = None,
    ) -> History:
        """训练模型并返回各轮训练和验证指标。

        参数:
            x: 按特征排列的训练数组列表或名称到数组的映射。
            y: 训练标签数组。
            batch_size: 每次梯度更新使用的样本数，默认 256。
            epochs: 训练结束轮次。
            verbose: 输出级别，0 表示静默，1 表示进度条，2 表示逐轮输出。
            initial_epoch: 起始轮次，用于继续训练。
            validation_split: 从训练数据末尾划出的验证集比例。
            validation_data: ``(x, y)`` 或 ``(x, y, sample_weight)`` 验证数据。
            shuffle: 是否在每轮开始前打乱数据。
            callbacks: 训练期间调用的回调列表。

        返回:
            记录训练和验证指标的 ``History`` 对象。
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    "When passing a `validation_data` argument, "
                    "it must contain either 2 items (x_val, y_val), "
                    "or 3 items (x_val, y_val, val_sample_weights), "
                    "or alternatively it could be a dataset or a "
                    "dataset or a dataset iterator. "
                    "However we received `validation_data=%s`" % validation_data
                )
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0.0 < validation_split < 1.0:
            do_validation = True
            if hasattr(x[0], "shape"):
                split_at = int(x[0].shape[0] * (1.0 - validation_split))
            else:
                split_at = int(len(x[0]) * (1.0 - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at), slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at), slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y)
        )
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            logger.info("parallel running on these gpus: {}", self.gpus)
            model = nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            logger.info(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size
        )

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, "model"):  # for tf1.4
            callbacks.__setattr__("model", self)
        callbacks.model.stop_training = False

        # Train
        logger.info(
            "Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
                len(train_tensor_data), len(val_y), steps_per_epoch
            )
        )
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        if isinstance(loss_func, list):
                            assert len(loss_func) == self.num_tasks, (
                                "the length of `loss_func` should be equal with `self.num_tasks`"
                            )
                            loss = sum(
                                [
                                    loss_func[i](y_pred[:, i], y[:, i], reduction="sum")
                                    for i in range(self.num_tasks)
                                ]
                            )
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction="sum")
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(
                                    metric_fun(
                                        y.cpu().data.numpy(),
                                        y_pred.cpu().data.numpy().astype("float64"),
                                    )
                                )

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                logger.info("Epoch {0}/{1}".format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"]
                )

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += (
                            " - "
                            + "val_"
                            + name
                            + ": {0: .4f}".format(epoch_logs["val_" + name])
                        )
                logger.success(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(
        self,
        x: list[np.ndarray] | dict[str, np.ndarray],
        y: np.ndarray,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """使用测试数据计算已配置的指标。

        参数:
            x: 按特征排列的测试数组列表或名称到数组的映射。
            y: 测试标签数组。
            batch_size: 每个评估批次的样本数。

        返回:
            指标名称到指标值的映射。
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(
        self,
        x: list[np.ndarray] | dict[str, np.ndarray],
        batch_size: int = 256,
    ) -> np.ndarray:
        """分批计算输入数据的预测值。

        参数:
            x: 按特征排列的输入数组列表或名称到数组的映射。
            batch_size: 每个预测批次的样本数。

        返回:
            模型预测值数组。
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size
        )

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def input_from_feature_columns(
        self,
        X: torch.Tensor,
        feature_columns: list[Any],
        embedding_dict: nn.ModuleDict,
        support_dense: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """从输入张量提取稀疏嵌入和连续特征值。"""
        sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )
        dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        varlen_sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
            if feature_columns
            else []
        )

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError("DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [
            embedding_dict[feat.embedding_name](
                X[
                    :,
                    self.feature_index[feat.name][0] : self.feature_index[feat.name][1],
                ].long()
            )
            for feat in sparse_feature_columns
        ]

        sequence_embed_dict = varlen_embedding_lookup(
            X, self.embedding_dict, self.feature_index, varlen_sparse_feature_columns
        )
        varlen_sparse_embedding_list = get_varlen_pooling_list(
            sequence_embed_dict,
            X,
            self.feature_index,
            varlen_sparse_feature_columns,
            self.device,
        )

        dense_value_list = [
            X[:, self.feature_index[feat.name][0] : self.feature_index[feat.name][1]]
            for feat in dense_feature_columns
        ]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(
        self,
        feature_columns: list[Any],
        include_sparse: bool = True,
        include_dense: bool = True,
        feature_group: bool = False,
    ) -> int:
        """计算指定特征列形成的模型输入维度。"""
        sparse_feature_columns = (
            list(
                filter(
                    lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)),
                    feature_columns,
                )
            )
            if len(feature_columns)
            else []
        )
        dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(
                feat.embedding_dim for feat in sparse_feature_columns
            )
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(
        self, weight_list: Any, l1: float = 0.0, l2: float = 0.0
    ) -> None:
        """登记需要计算 L1 或 L2 正则损失的参数。"""
        # 将单个参数包装为列表，以兼容正则损失的统一处理
        if isinstance(weight_list, nn.parameter.Parameter):
            weight_list = [weight_list]
        # 将生成器和参数列表实体化，避免保存模型时无法序列化
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(
        self,
    ) -> torch.Tensor:
        """计算已登记参数的正则损失。"""
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss: torch.Tensor, alpha: float) -> None:
        """设置按给定系数缩放的辅助损失。"""
        self.aux_loss = aux_loss * alpha

    def compile(
        self,
        optimizer: str | torch.optim.Optimizer,
        loss: str | list[str] | Any | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        """配置训练使用的优化器、损失函数和评估指标。

        参数:
            optimizer: 优化器名称或 PyTorch 优化器实例。
            loss: 损失函数名称、名称列表或可调用对象。
            metrics: 训练和测试期间计算的指标名称列表。

        返回:
            无返回值。
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [
                self._get_loss_func_single(loss_single) for loss_single in loss
            ]
        else:
            loss_func = loss
        return loss_func

    def _get_loss_func_single(self, loss):
        if loss == "binary_crossentropy":
            loss_func = F.binary_cross_entropy
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

    def _log_loss(
        self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None
    ):
        # change eps to improve calculation accuracy
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(
        self,
    ) -> int:
        """返回所有稀疏特征共用的嵌入维度。"""
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = (
            list(
                filter(
                    lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)),
                    feature_columns,
                )
            )
            if len(feature_columns)
            else []
        )
        embedding_size_set = set(
            [feat.embedding_dim for feat in sparse_feature_columns]
        )
        if len(embedding_size_set) > 1:
            raise ValueError(
                "embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!"
            )
        return list(embedding_size_set)[0]
