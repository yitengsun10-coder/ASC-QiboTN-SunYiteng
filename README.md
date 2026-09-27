# ASC QiboTN CPU 量子线路模拟优化记录

> **最新正式结果：** 本仓库保留早期 16/18/20-qubit 实验作为历史与
> 环境验证。按照后续“大规模 workload”要求完成的 30 qubits、9,998
> gates、depth 449 官方 CPU 路径实验，已迁移到官方源码 fork：
> [yitengsun10-coder/qibotn](https://github.com/yitengsun10-coder/qibotn/tree/main/benchmarks/asc_large_cpu)。
> 该版本含原始日志、CSV、环境、保真度验证和可复现脚本；正式有效优化为
> 相同数值配置下 16→1 线程，稳态 324.6081 s→57.5065 s（5.6447×）。

- 学生：孙逸腾（240810010427）
- 题目：QiboTN
- 官方文档：https://qibo.science/qibotn/stable/
- 官方仓库：https://github.com/qiboteam/qibotn
- 工作负载：6 层 QAOA，16/18/20 qubit，CPU
- 本仓库早期最佳方案：`mps_cutoff_1e8`，稳态均值 0.39196 s，相对 baseline 2.3716×，最低保真度 0.9999999999999886

本仓库包含两个可直接从终端调用的 Python benchmark、依赖文件、两批原始 CSV、资源日志、环境和汇总 JSON。

## 安装

使用 Python 3.11 和独立虚拟环境。实测依赖中的 `qibojit 0.1.15` 不支持 Python 3.12，因此不要用 3.12 创建该环境：

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps \
  "git+https://github.com/qiboteam/qibotn.git@4775e50c4c19a00153d8e8700ec2785b9f9ef3e1"
```

QiboTN 0.0.7 尚未以该版本发布到 PyPI，因此固定官方源码提交并使用 `--no-deps`，避免当前项目元数据额外拉取与 CPU 实验无关的 CUDA 工具包。该安装方式已在 Windows + Python 3.11.16 上实际跑通 8-qubit 快速实验。本次正式实测版本为 qibo 0.3.2、qibojit 0.1.15、qibotn 0.0.7、quimb 1.13.0，详见 [`environment.log`](environment.log)。

## 直接运行

先用小规模确认环境和输出目录：

```bash
python qibotn_benchmark_v2.py --output reproduced/quick --qubits 8 10 --repeats 2
```

在 Windows PowerShell 中同样直接使用上述 `python` 命令，不需要 Bash。

复现实验规模：

```bash
python qibotn_benchmark_v2.py --output reproduced/formal --qubits 16 18 20 --repeats 4
```

脚本会生成 `results.csv`、`summary.json` 和 `environment.json`。输出目录必须显式指定，避免覆盖仓库内原始证据。

## 一分钟证据验收

无需安装 QiboTN，只需 Python 标准库：

```bash
python tools/verify_evidence.py
```

该脚本检查 `small/` 和 `large/` 的 CSV/JSON、所有状态、误差、保真度及 SHA-256 清单。

## 优化设计

基准与优化保持同一 QAOA 线路定义和随机输入，只改变张量网络执行策略：

1. `baseline_tn`：基础张量网络收缩。
2. `mps_exact`：精确 MPS，对照算法切换本身的开销。
3. `mps_cutoff_1e12`：较保守截断，观察速度和精度。
4. `mps_cutoff_1e8`：更积极截断，仍用 dense reference 验证误差和保真度。
5. 固定 BLAS/线程池线程数并区分冷启动与稳态时间，避免把初始化开销误判为内核性能。

## 结果摘要

| 数据组 | 方法 | 稳态均值 s | 相对 baseline | 最大绝对误差 | 最低保真度 |
|---|---|---:|---:|---:|---:|
| small | baseline_tn | 0.12708 | 1.0000× | 8.78e-15 | 0.9999999999999997 |
| small | mps_cutoff_1e8 | 0.12187 | 1.0427× | 1.80e-9 | 0.9999999999999971 |
| large | baseline_tn | 0.92955 | 1.0000× | 3.98e-15 | 0.9999999999999980 |
| large | mps_cutoff_1e12 | 0.89122 | 1.0430× | 1.53e-13 | 0.9999999999999925 |
| large | mps_cutoff_1e8 | **0.39196** | **2.3716×** | 2.41e-9 | 0.9999999999999886 |

`mps_exact` 在大规模组平均 26.50 s，明显慢于 baseline，说明算法名称“精确”不等于更高性能；负结果也保留在原始数据中。

## 原始证据

- [`small/results.csv`](small/results.csv)、[`small/summary.json`](small/summary.json)、[`small/run.log`](small/run.log)、[`small/resource.log`](small/resource.log)
- [`large/results.csv`](large/results.csv)、[`large/summary.json`](large/summary.json)、[`large/run.log`](large/run.log)、[`large/resource.log`](large/resource.log)
- [`qibotn_benchmark_v2.py`](qibotn_benchmark_v2.py)：最终 benchmark 入口。
- [`qibotn_benchmark.py`](qibotn_benchmark.py)：早期对照入口，原样保留。
- [`SHA256SUMS.txt`](SHA256SUMS.txt)：当前仓库副本的可直接校验哈希。
- [`ARCHIVED_SERVER_SHA256SUMS.txt`](ARCHIVED_SERVER_SHA256SUMS.txt)：原始服务器绝对路径清单；两份 CSV 与当前副本一致，两个 JSON 曾因整理时格式化而字节哈希变化，数值由校验器逐字段复核。

所有正式结果来自 CPU；没有用 GPU 或修改线路定义降低任务难度。
