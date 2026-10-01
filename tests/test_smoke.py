"""funrec 公共 API 的轻量测试。

funrec has no ``[project.scripts]`` CLI entry point, so this suite focuses on:
  * importability of the top-level package and its public submodules
  * constructing/exercising a few representative public classes with trivial,
    synthetic inputs (no real data files, no network, no GPU)
  * making sure the (rarely used) network-touching helper ``check_version``
    never performs a real HTTP call during tests

回调模块使用 PyTorch 兼容实现，因此测试可以在没有 TensorFlow 的干净环境中导入。
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

import funrec
import funrec.callbacks
import funrec.inputs
import funrec.layers
import funrec.models
from funrec.inputs import (
    DenseFeat,
    SparseFeat,
    VarLenSparseFeat,
    build_input_features,
    get_feature_names,
)
from funrec.layers import DNN, PredictionLayer
from funrec.layers.utils import slice_arrays
from funrec.models import WDL, DeepFM


def test_top_level_package_imports():
    """`import funrec` must not raise, and its documented public API must exist."""
    assert hasattr(funrec, "layers")
    assert hasattr(funrec, "models")
    assert hasattr(funrec, "check_version")
    assert set(funrec.__all__) == {"layers", "models", "check_version"}


def test_public_submodules_import_cleanly():
    for name in funrec.layers.__all__:
        assert hasattr(funrec.layers, name)
    for name in funrec.inputs.__all__:
        assert hasattr(funrec.inputs, name)
    for name in funrec.models.__all__:
        assert hasattr(funrec.models, name)


def test_feature_columns_and_input_index():
    """SparseFeat/DenseFeat/VarLenSparseFeat + build_input_features is the core
    public API every model in funrec.models is built on top of."""
    sparse = SparseFeat("user_id", vocabulary_size=10, embedding_dim=4)
    dense = DenseFeat("price", dimension=1)
    varlen = VarLenSparseFeat(
        SparseFeat("hist_item", vocabulary_size=10, embedding_dim=4), maxlen=5
    )

    feature_columns = [sparse, dense, varlen]
    feature_index = build_input_features(feature_columns)

    assert list(feature_index.keys()) == ["user_id", "price", "hist_item"]
    assert get_feature_names(feature_columns) == ["user_id", "price", "hist_item"]


def test_build_input_features_rejects_unknown_feature_column():
    """特征索引构建器应明确拒绝不支持的特征列类型。"""
    with pytest.raises(TypeError):
        build_input_features([object()])


def test_slice_arrays_handles_none_and_rejects_ambiguous_indices():
    """数组切片工具应覆盖空输入，并拒绝同时传入索引列表和结束位置。"""
    assert slice_arrays(None) == [None]
    with pytest.raises(ValueError):
        slice_arrays([torch.arange(3)], [0, 1], 2)


def test_dnn_and_prediction_layer_forward():
    """Exercise a couple of low-level building blocks with a tiny random batch."""
    dnn = DNN(inputs_dim=8, hidden_units=(16, 4))
    prediction_layer = PredictionLayer(task="binary")

    x = torch.randn(3, 8)
    hidden = dnn(x)
    assert hidden.shape == (3, 4)

    logit = torch.randn(3, 1)
    pred = prediction_layer(logit)
    assert pred.shape == (3, 1)
    assert torch.all((pred >= 0) & (pred <= 1))


def test_prediction_layer_rejects_unknown_task():
    """预测层应拒绝未定义的任务类型。"""
    with pytest.raises(ValueError):
        PredictionLayer(task="unknown")


@pytest.mark.parametrize("model_cls", [WDL, DeepFM])
def test_ctr_model_construct_and_forward(model_cls):
    """Build a tiny WDL/DeepFM model from synthetic feature columns and run a
    forward pass -- no real dataset, no training, just a shape/sanity check
    that the public model API works end to end."""
    sparse_feature_columns = [
        SparseFeat("user_id", vocabulary_size=4, embedding_dim=4),
        SparseFeat("item_id", vocabulary_size=4, embedding_dim=4),
    ]
    dense_feature_columns = [DenseFeat("price", dimension=1)]
    feature_columns = sparse_feature_columns + dense_feature_columns

    model = model_cls(
        linear_feature_columns=feature_columns,
        dnn_feature_columns=feature_columns,
        dnn_hidden_units=(8, 4),
        device="cpu",
    )

    feature_index = build_input_features(feature_columns)
    batch_size = 5
    x = torch.zeros(batch_size, len(feature_index))
    for name, (start, end) in feature_index.items():
        if name in ("user_id", "item_id"):
            x[:, start:end] = torch.randint(0, 4, (batch_size, 1))
        else:
            x[:, start:end] = torch.randn(batch_size, end - start)

    with torch.no_grad():
        y_pred = model(x)

    assert y_pred.shape == (batch_size, 1)
    assert torch.all((y_pred >= 0) & (y_pred <= 1))


def test_check_version_never_makes_a_real_network_call():
    """`funrec.utils.check_version` spawns a background thread that calls
    `requests.get` against pypi.python.org. We must never let a smoke test hit
    the real network, so we replace Thread with a synchronous stand-in and
    mock `requests.get`."""
    from funrec import utils

    class ImmediateThread:
        def __init__(self, target=None, args=(), **kwargs):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.text = json.dumps({"releases": {}})
    fake_response.raise_for_status.return_value = None

    with (
        patch.object(utils, "Thread", ImmediateThread),
        patch.object(utils.requests, "get", return_value=fake_response) as mock_get,
    ):
        utils.check_version("1.0.0")
        mock_get.assert_called_once_with(
            "https://pypi.org/pypi/deepctr-torch/json", timeout=5
        )


def test_check_version_handles_request_failure():
    """网络失败必须在后台任务内被记录，且不泄漏为未处理异常。"""
    from funrec import utils

    class ImmediateThread:
        def __init__(self, target=None, args=(), **_kwargs):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    with (
        patch.object(utils, "Thread", ImmediateThread),
        patch.object(
            utils.requests,
            "get",
            side_effect=utils.requests.Timeout("timed out"),
        ),
        patch.object(utils.logger, "error") as log_error,
    ):
        utils.check_version("1.0.0")
        assert "检查 deepctr-torch 版本失败" in log_error.call_args.args[0]


def test_check_version_handles_invalid_response():
    """版本接口返回非对象 JSON 时应记录解析上下文并结束后台任务。"""
    from funrec import utils

    class ImmediateThread:
        def __init__(self, target=None, args=(), **_kwargs):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    fake_response = MagicMock()
    fake_response.text = "[]"
    fake_response.raise_for_status.return_value = None

    with (
        patch.object(utils, "Thread", ImmediateThread),
        patch.object(utils.requests, "get", return_value=fake_response),
        patch.object(utils.logger, "error") as log_error,
    ):
        utils.check_version("1.0.0")
        assert "解析 deepctr-torch 版本响应失败" in log_error.call_args.args[0]


def test_mind_train_import_does_not_read_data():
    """导入训练模块不得读取 CSV 或启动训练。"""
    import importlib

    with patch("pandas.read_csv") as read_csv:
        module = importlib.import_module("funrec.models.p2019.mind.train")
        assert callable(module.main)
        read_csv.assert_not_called()


def test_callbacks_module_importable_and_defines_expected_names():
    """funrec.callbacks 在没有 TensorFlow 的环境中也应提供公开名称。"""
    assert hasattr(funrec.callbacks, "ModelCheckpoint")
    assert hasattr(funrec.callbacks, "History")
    assert hasattr(funrec.callbacks, "CallbackList")
