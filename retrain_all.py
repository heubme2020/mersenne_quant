"""兼容垫片：本脚本的功能已于 2026-09-28 合并进 `refresh_model.py`（三个脚本三合一）。

## 为什么还留着

**2026-09-28 07:57 启动的那次一键重训正在跑**，它的阶段 3 会在几小时后以
`python retrain_all.py [--only ...]` 的形式调用本文件（那个父进程读到的是旧代码、
改不了）。现在删掉它，那次重训就会在训练段崩、白跑十几小时。

## 等价入口

    python refresh_model.py --stage train [--only one two] [--dry-run]

等那次重训跑完，本文件即可删除（它不是任何逻辑的实现，只是转发）。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
cmd = [sys.executable, os.path.join(ROOT, 'refresh_model.py'), '--stage', 'train', *sys.argv[1:]]
sys.exit(subprocess.call(cmd, cwd=ROOT))
