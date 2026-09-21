import torch
from farlog import getLogger

try:
    from tensorflow.python.keras.callbacks import CallbackList, EarlyStopping, History
    from tensorflow.python.keras.callbacks import ModelCheckpoint as _ModelCheckpoint
except ImportError:
    class _Callback:
        """PyTorch 项目使用的轻量回调基类。"""

        def __init__(self, *args, **kwargs):
            self.model = None

        def set_model(self, model):
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

        def __init__(self, filepath, monitor="val_loss", verbose=0,
                     save_best_only=False, mode="auto", save_weights_only=False,
                     period=1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.filepath = filepath
            self.monitor = monitor
            self.verbose = verbose
            self.save_best_only = save_best_only
            self.save_weights_only = save_weights_only
            self.period = period
            self.epochs_since_last_save = 0
            self.best = float("inf") if mode == "min" else float("-inf")
            self.monitor_op = (lambda current, best: current < best) if mode == "min" else (lambda current, best: current > best)

from funrec.layers import DNN

__all__ = ["DNN", "ModelCheckpoint", "History", "EarlyStopping", "CallbackList"]

logger = getLogger("funrec")


class ModelCheckpoint(_ModelCheckpoint):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def on_epoch_end(self, epoch, logs=None):
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
