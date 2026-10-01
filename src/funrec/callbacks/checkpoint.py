from typing import Any

import torch
from farlog import getLogger

try:
    from tensorflow.python.keras.callbacks import CallbackList, EarlyStopping, History
    from tensorflow.python.keras.callbacks import ModelCheckpoint as _ModelCheckpoint
except ImportError:

    class _Callback:
        """PyTorch 项目使用的轻量回调基类。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.model = None

        def set_model(self, model: torch.nn.Module) -> None:
            """设置回调所操作的模型。"""
            self.model = model

    class CallbackList(list):
        """兼容 Keras 名称的回调列表。"""

    class EarlyStopping(_Callback):
        """兼容 Keras 名称的提前停止回调占位实现。"""

    class History(_Callback):
        """记录训练历史的轻量回调。"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.history = {}

    class _ModelCheckpoint(_Callback):
        """提供 ModelCheckpoint 所需的最小配置接口。"""

        def __init__(
            self,
            filepath: str,
            monitor: str = "val_loss",
            verbose: int = 0,
            save_best_only: bool = False,
            mode: str = "auto",
            save_weights_only: bool = False,
            period: int = 1,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.filepath = filepath
            self.monitor = monitor
            self.verbose = verbose
            self.save_best_only = save_best_only
            self.save_weights_only = save_weights_only
            self.period = period
            self.epochs_since_last_save = 0
            self.best = float("inf") if mode == "min" else float("-inf")
            self.monitor_op = (
                (lambda current, best: current < best)
                if mode == "min"
                else (lambda current, best: current > best)
            )


from funrec.layers import DNN

__all__ = ["DNN", "ModelCheckpoint", "History", "EarlyStopping", "CallbackList"]

logger = getLogger("funrec")


class ModelCheckpoint(_ModelCheckpoint):
    """在训练轮次结束时按条件保存模型检查点。

    参数:
        filepath: 保存路径模板，可引用 ``epoch`` 和日志中的指标名称。
        monitor: 用于判断最佳模型的指标名称。
        verbose: 输出级别，取 0 或 1。
        save_best_only: 是否仅保存监控指标更优的模型。
        mode: 指标比较方式，支持 ``auto``、``min`` 和 ``max``。
        save_weights_only: 是否仅保存模型权重。
        period: 两次检查点保存之间的训练轮数。

    返回:
        初始化后的检查点回调对象。
    """

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """处理轮次结束事件，并在满足条件时写入检查点。"""
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logger.info(
                        "Can save best model only with %s available, skipping."
                        % self.monitor
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            logger.info(
                                "Epoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.best = current
                        if self.save_weights_only:
                            torch.save(self.model.state_dict(), filepath)
                        else:
                            torch.save(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            logger.success(
                                f"Epoch {epoch + 1:05d}: {self.monitor} did not improve from {self.best:0.5f}"
                            )
            else:
                if self.verbose > 0:
                    logger.success(
                        "Epoch %05d: saving model to %s" % (epoch + 1, filepath)
                    )
                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save(self.model, filepath)
