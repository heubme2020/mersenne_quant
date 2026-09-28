"""兼容垫片：让**现有**的生产 `two/two.pt` 在没有 `nowcast/` 的环境里也能 `torch.load`。

## 为什么必须有它（2026-09-27 发现的真实断点）

现有 `two/two.pt` 是 `two/nowcast/train.py` 用 `torch.save(model)` 存的，pickle 里记的类路径是
**`model.Nowcast`**（即 `two/nowcast/model.py` 里的那个类，实测 `two.pt` 的 pickle 头就是
`cmodel\nNowcast\n`）。而 `two/get_two_predict.py` 只把 `../one` 加进 `sys.path`，
**没有** nowcast/ —— 于是：

    >>> torch.load('two/two.pt', weights_only=False)
    ModuleNotFoundError: No module named 'model'

也就是说："two 不再依赖 nowcast/" 这件事，光把**特征**搬出来还不够，**权重文件本身**
还拴着 two/nowcast/model.py。本文件把 `model` 这个名字指到 `two_nowcast_model`（逐字复制的那份）
——pickle 解析 `model.Nowcast` 时就能拿到同一个类，`two/` 从此自足。

## 以后不再需要它

`two/train.py` 存出来的新 `two.pt`，pickle 记的是 `two_nowcast_model.Nowcast`，
不需要这个垫片。留着无害（`two/` 之外没人会 import 一个叫 `model` 的顶层模块，
且本文件只是转发，没有新代码）。
"""
from two_nowcast_model import (Nowcast, RegressionHead, set_output_scale,   # noqa: F401
                              N_HEADS, FLAT_NAMES, FIELDS, AUX_OUTPUT_DAYS)
