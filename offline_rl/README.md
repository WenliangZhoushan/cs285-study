# 离线强化学习：IQL 与 CQL（学习版）

这是两个常见离线强化学习算法在 `Pendulum-v1` 上的最小标准实现：IQL 和
CQL。代码优先保证清晰易读，而不是追求训练速度或工程完整性。

## 文件结构

```text
offline_rl/
├── networks.py       # TanhGaussianActor、TwinQ、ValueCritic
├── replay_buffer.py  # 基于 numpy 的 replay buffer
├── utils.py          # set_seed、soft_update、evaluate
├── collect_data.py   # 生成离线数据集（启发式专家 + 随机策略混合）
├── iql.py            # IQL agent
├── cql.py            # CQL agent（连续动作版 SAC-CQL）
├── train_iql.py      # 在离线数据上训练 IQL
└── train_cql.py      # 在离线数据上训练 CQL
```

## 快速开始

请从仓库根目录运行下面的命令，这样 Python 才能正确解析 `offline_rl` 包：

```bash
# 1) 生成约 5 万条离线 transition（主要来自带噪声专家，少量来自随机策略）
python -m offline_rl.collect_data --n 50000 --expert-frac 0.7

# 2) 训练 IQL
python -m offline_rl.train_iql --steps 50000

# 3) 训练 CQL
python -m offline_rl.train_cql --steps 50000
```

`Pendulum-v1` 上随机策略每个 episode 的回报大约是 **-1200**；这里用于采集
数据的启发式策略质量非常混合（最好约 -1，最差约 -1900，平均约 -1280）。
两个 agent 在大约 1 万次更新内都应该明显优于行为策略。

本仓库的 smoke test 结果如下（CPU、默认参数、3 万条离线 transition）：

| 算法 | step 2k | step 4k | step 6k | step 10k |
|------|---------|---------|---------|----------|
| IQL  | -1128   | -290    | -897    | -194     |
| CQL  | -1083   | -237    | -205    | —        |

IQL 训练 1 万步大约需要 120 秒；CQL 因为要计算 N-sample `logsumexp`，训练
1 万步大约需要 9 分钟。如果只想快速调试，可以减小 `--cql-n-samples`。

## 调试配置

仓库的 `.vscode/launch.json` 里已经加入了三个 `offline_rl` 调试配置：

- `offline_rl: collect data (debug)`：生成较小的调试数据集
  `offline_rl/data/pendulum_debug.npz`。
- `offline_rl: train IQL (debug)`：用调试数据集训练 IQL，默认只跑 2000 步。
- `offline_rl: train CQL (debug)`：用调试数据集训练 CQL，默认只跑 2000 步，并将
  `--cql-n-samples` 降到 4 以加快单步调试。

建议先运行数据采集配置，再运行 IQL 或 CQL 的训练配置。所有配置都以仓库根目录
作为 `cwd`，等价于使用 `python -m offline_rl...` 启动脚本。

## 算法概要

### IQL（Kostrikov et al., 2021）

IQL 有三个损失，全部在离线 buffer 中的 `(s, a, r, s')` 上计算：

- `V(s)`：对 `Q_target(s, a)` 做 expectile regression，即
  `L_tau(Q_target - V)`。
- `Q(s,a)`：TD 回归，即 `(Q - (r + gamma * V(s')))^2`。
- `pi(a|s)`：AWR，即 `- exp(beta * (Q_target - V)) * log pi(a|s)`。

核心思想是：`V` 被拟合成数据分布下 `Q` 的 tau-expectile，因此它能近似
`max_a Q(s,a)`，同时**不需要在离线数据之外的动作上评估 Q**。

### CQL（Kumar et al., 2020）

CQL 可以理解为标准 SAC 加上 critic 的保守性惩罚：

```text
L_critic = SAC_TD_loss
         + alpha_cql * (logsumexp_a Q(s, a)  -  Q(s, a_data))
```

其中 `logsumexp_a Q(s, a)` 使用来自 `{Uniform, π(·|s), π(·|s')}` 的重要性采样
近似计算（见 CQL 论文 Appendix F）。Actor 和 entropy temperature 仍使用未修改的
SAC 目标。

## 说明与取舍

- 所有网络都是 2x256 MLP；观测维度和动作维度会从环境中自动推断。
- 这里的 `done` 表示 **terminal**，不表示 truncation。`Pendulum-v1` 没有真正的
  terminal，因此 bootstrap target 总是会被使用。
- 这些实现省略了完整 RL 库里常见的一些工程增强，例如 `alpha_cql` 的 Lagrange
  自动调节、layer norm、gradient clipping、learning-rate scheduler。如果需要更强
  的结果，可以在此基础上继续添加。
