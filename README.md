# funrec

`funrec` 是基于 PyTorch 的深度学习推荐模型与特征工程工具集，提供 CTR 模型、特征列和常用网络层。

## 安装
```bash
pip install funrec
```

## 使用

最小示例：

```python
import torch
from funrec.inputs import DenseFeat, SparseFeat, build_input_features
from funrec.models import DeepFM

columns = [SparseFeat("user", 4, embedding_dim=4), DenseFeat("score")]
model = DeepFM(columns, columns, dnn_hidden_units=(8, 4))
features = build_input_features(columns)
x = torch.zeros(2, len(features))
print(model(x).shape)  # torch.Size([2, 1])
```

更多训练示例见 `examples/`。


## 感恩的心

*  [DeepCTR](https://zhuanlan.zhihu.com/p/53231955)  易用可扩展的深度学习点击率预测算法包。
    * [DeepCTR](https://github.com/shenweichen/DeepCTR.git)：上游项目采用 MIT License。
    * [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch.git)：上游项目采用 MIT License，本项目部分代码来源于此。
* *fun-rec* 推荐系统的整体介绍，包含 推荐系统概述、推荐算法基础、推荐系统实战和推荐系统面经
    * [文档](https://datawhalechina.github.io/fun-rec/#/)
    * [源码](https://github.com/datawhalechina/fun-rec.git)

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
