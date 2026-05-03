# Offline RL: IQL & CQL (learning-oriented)

Minimal, standard implementations of two popular offline RL algorithms on
Pendulum-v1, written for clarity rather than performance.

## Files

```
offline_rl/
├── networks.py       # TanhGaussianActor, TwinQ, ValueCritic
├── replay_buffer.py  # numpy-backed buffer
├── utils.py          # set_seed, soft_update, evaluate
├── collect_data.py   # generates offline dataset (mixed expert + random)
├── iql.py            # IQL agent
├── cql.py            # CQL agent (continuous SAC-CQL)
├── train_iql.py      # train IQL on offline data
└── train_cql.py      # train CQL on offline data
```

## Quickstart

Run from the repo root (so the `offline_rl` package resolves):

```bash
# 1) generate ~50k offline transitions (mostly noisy expert + some random)
python -m offline_rl.collect_data --n 50000 --expert-frac 0.7

# 2) train IQL
python -m offline_rl.train_iql --steps 50000

# 3) train CQL
python -m offline_rl.train_cql --steps 50000
```

A random policy on Pendulum-v1 returns roughly **-1200** per episode; the
heuristic used to generate data has highly mixed quality (best episodes
~ -1, worst ~ -1900, mean ~ -1280). Both agents should clearly beat the
behavior policy within ~10k updates.

Smoke-test results from this repo (CPU, default args, 30k offline transitions):

| algorithm | step 2k | step 4k | step 6k | step 10k |
|-----------|---------|---------|---------|----------|
| IQL       | -1128   | -290    | -897    | -194     |
| CQL       | -1083   | -237    | -205    |  —       |

(IQL ~120s for 10k steps; CQL ~9 min for 10k steps because of the
N-sample logsumexp. Reduce `--cql-n-samples` for faster runs.)

## Algorithm summaries

### IQL (Kostrikov et al., 2021)

Three losses, all evaluated on `(s, a, r, s')` from the offline buffer:

| network | objective |
|---------|-----------|
| `V(s)`  | expectile regression of `Q_target(s, a)`: `L_tau(Q_target - V)` |
| `Q(s,a)` | TD regression: `(Q - (r + gamma * V(s')))^2` |
| `pi(a|s)` | AWR: `- exp(beta * (Q_target - V)) * log pi(a|s)` |

Key idea: `V` is fit as the τ-expectile of `Q` over the data distribution, so
it approximates `max_a Q(s,a)` *without ever evaluating Q at unseen actions*.

### CQL (Kumar et al., 2020)

Standard SAC + a conservatism penalty on the critic:

```
L_critic = SAC_TD_loss
         + alpha_cql * (logsumexp_a Q(s, a)  -  Q(s, a_data))
```

`logsumexp_a Q(s, a)` is approximated with importance sampling from
{Uniform, π(·|s), π(·|s')} (CQL paper, Appendix F). The actor and entropy
temperature use the unmodified SAC objectives.

## Notes / tradeoffs

- Networks are 2x256 MLPs everywhere; obs/action dims are inferred from the
  environment.
- `done` here means **terminal**, not truncation — Pendulum has no real
  terminal, so the bootstrap target is always used.
- These implementations skip a few engineering refinements you would find in
  a full library (Lagrange auto-tuning of `alpha_cql`, layer norm, gradient
  clipping, learning-rate schedulers). Add them if you want stronger results.
