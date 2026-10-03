# 更新日志

## [未发布]

### 新增

- `pyproject.toml` 新增 `examples`（`pandas`）与 `mind`（`faiss-cpu`、`matplotlib`、`pandas`）两个可选依赖分组，并在 README 说明安装方式；`numpy` 补充为直接依赖。
- `tests/test_smoke.py` 大幅扩充模型覆盖范围：新增 CCPM、AFM、AFN、MLR、DCN、DCNMix、NFM、XDeepFM、AutoInt、FiBiNet、IFM、ONN、DIFM、PNN、DIN、DIEN、MMOE、ESMM、PLE、SharedBottom、MIND 的构造/前向 smoke test，以及非法 task、空 `dnn_hidden_units` 等边界用例。

### 修复

- 修复 `DIEN` 默认 `gru_type="GRU"` 分支中 `nn.GwRU` 拼写错误（应为 `nn.GRU`），此前默认配置下构造模型即崩溃。
- 修复 `MMOE`/`PLE` 门控输出 `gate_mul_expert.squeeze()` 未指定维度，在 `batch_size == 1` 时会错误压缩掉批次维度。
- 修复 `ONN` 的 `Interac.__init_weight` 只初始化了 `emb1.weight`，遗漏 `emb2.weight`。
- 修复 `COMI`（`p2020/comi`）胶囊网络中误用 PaddlePaddle 专有的 `self.create_parameter`，PyTorch 下会直接抛出 `AttributeError`；改为 `nn.Parameter` + `xavier_normal_` 初始化。
- 修复 `build_input_features` 在遇到不支持的特征列类型时，先访问 `feat.name` 导致抛出难以定位的 `AttributeError`，而不是文档约定的 `TypeError`；现在先校验类型再取名。
- `src/funrec/utils.py` 的版本检查不再查询上游 `deepctr-torch` 的 PyPI 元数据，改为查询 `funrec` 自身版本，日志与升级提示同步更新。
- `examples/` 下各脚本按脚本自身路径解析同目录数据文件，不再依赖运行时工作目录；`run_multivalue_movielens.py` 移除对 `keras` 的隐式依赖，改用本地最小 `pad_sequences` 实现。
- `mind/train.py` 的 `print` 诊断信息改为 `farlog` 结构化日志；`MIND(active_config)` 的构造签名与类定义不一致问题予以修正（改为显式关键字参数）。

### 变更

- `src/funrec/models/` 下 22 个模型 `code.py`、`b2000/line.py`、`p2019/mind/core.py` 补齐中文 docstring 与 Python 3.10 风格类型标注（涵盖 `__init__`/`forward` 及关键私有方法）。
- README 补充示例安装方式（`pip install funrec[examples]`/`funrec[mind]`）与运行方式说明。

### 废弃

- 无。

## [0.0.9] - 2026-09-21

### 新增

- 增加推荐模型、特征列和回调的最小可运行示例。

### 修复

- 移除对 TensorFlow 内部回调模块的强制导入，改为 PyTorch 可用的兼容实现。
- 改用公开的 `packaging` 依赖，避免使用 pip 内部模块。
- 版本检查增加请求超时，并分别记录网络、响应和版本解析错误。
- MIND 训练脚本导入时不再读取数据或启动训练。

### 变更

- 补齐 README 上游许可证说明和运行环境忽略规则。

### 废弃

- 无。
