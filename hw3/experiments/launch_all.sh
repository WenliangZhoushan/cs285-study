#!/usr/bin/env bash
# Launch all hw3 DQN / SAC training runs, packing multiple jobs per GPU.
#
# Usage:
#   bash experiments/launch_all.sh                 # use defaults
#   GPUS="4 5 6 7" JOBS_PER_GPU=3 bash ...         # override scheduling
#   SEEDS="1 2 3" bash ...                         # override seeds
#   INCLUDE_LONG=0 bash ...                        # drop million-step runs
#   WANDB_MODE=offline bash ...                    # offline wandb
#   DRY_RUN=1 bash ...                             # print, don't launch
#
# The launcher maintains a fixed-size pool of slots (GPUS x JOBS_PER_GPU). It
# fills the pool from a queue of (script, config, seed) tasks; when any slot
# frees up, it launches the next pending task on that slot's GPU.

set -u

# ---------- Paths ----------
HW3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HW3_DIR"

SCRIPTS_DIR="$HW3_DIR/cs285/scripts"
RUN_DQN="$SCRIPTS_DIR/run_hw3_dqn.py"
RUN_SAC="$SCRIPTS_DIR/run_hw3_sac.py"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$HW3_DIR/experiments/launch_logs/$TS}"
mkdir -p "$LOG_ROOT"

# ---------- Scheduling knobs ----------
GPUS=${GPUS:-"4 5 6 7"}
JOBS_PER_GPU=${JOBS_PER_GPU:-3}
SEEDS=${SEEDS:-"1 2 3"}
INCLUDE_LONG=${INCLUDE_LONG:-1}    # 1 = include 1M-step halfcheetah/mspacman runs
INCLUDE_HUMANOID=${INCLUDE_HUMANOID:-0}   # 0 = skip 5M-step humanoid run

# ---------- wandb knobs ----------
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-}
DQN_PROJECT=${DQN_PROJECT:-cs285-hw3-dqn}
SAC_PROJECT=${SAC_PROJECT:-cs285-hw3-sac}

DRY_RUN=${DRY_RUN:-0}

# ---------- Conda env ----------
CONDA_ENV=${CONDA_ENV:-cs285}
if [[ -f /data/wmz5132/anaconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /data/wmz5132/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
fi

# ---------- Build the task list ----------
# Each task: ALGO|CFG_PATH|SEED  -> short DQN / sanity runs first (fast feedback),
# then bigger million-step jobs.

DQN_FAST_CFGS=(
    "experiments/dqn/cartpole.yaml"
    "experiments/dqn/lunarlander.yaml"
    "experiments/dqn/lunarlander_doubleq.yaml"
)
DQN_LONG_CFGS=(
    "experiments/dqn/mspacman.yaml"
)

SAC_SANITY_CFGS=(
    "experiments/sac/sanity_pendulum.yaml"
    "experiments/sac/sanity_invertedpendulum_reinforce.yaml"
    "experiments/sac/sanity_invertedpendulum_reparametrize.yaml"
)
SAC_HOPPER_CFGS=(
    "experiments/sac/hopper.yaml"
    "experiments/sac/hopper_doubleq.yaml"
    "experiments/sac/hopper_clipq.yaml"
)
SAC_HALFCHEETAH_CFGS=(
    "experiments/sac/halfcheetah_reparametrize.yaml"
    "experiments/sac/halfcheetah_reinforce1.yaml"
    "experiments/sac/halfcheetah_reinforce10.yaml"
    "experiments/sac/halfcheetah_clipq.yaml"
    "experiments/sac/halfcheetah_doubleq.yaml"
)
SAC_HUMANOID_CFGS=(
    "experiments/sac/humanoid.yaml"
)

declare -a TASKS
add_tasks() {
    local algo="$1"; shift
    for cfg in "$@"; do
        for seed in $SEEDS; do
            TASKS+=("$algo|$cfg|$seed")
        done
    done
}

# Short first, long last — so we see signal early in wandb.
add_tasks dqn "${DQN_FAST_CFGS[@]}"
add_tasks sac "${SAC_SANITY_CFGS[@]}"
add_tasks sac "${SAC_HOPPER_CFGS[@]}"

if [[ "$INCLUDE_LONG" == "1" ]]; then
    add_tasks sac "${SAC_HALFCHEETAH_CFGS[@]}"
    add_tasks dqn "${DQN_LONG_CFGS[@]}"
fi
if [[ "$INCLUDE_HUMANOID" == "1" ]]; then
    add_tasks sac "${SAC_HUMANOID_CFGS[@]}"
fi

echo "=================================================================="
echo "hw3 training launcher"
echo "  log dir      : $LOG_ROOT"
echo "  GPUS         : $GPUS"
echo "  JOBS_PER_GPU : $JOBS_PER_GPU"
echo "  SEEDS        : $SEEDS"
echo "  INCLUDE_LONG : $INCLUDE_LONG"
echo "  WANDB_MODE   : $WANDB_MODE"
echo "  num tasks    : ${#TASKS[@]}"
echo "=================================================================="

# ---------- Build the slot list (one GPU id per slot) ----------
declare -a SLOT_GPU
for g in $GPUS; do
    for ((i=0; i<JOBS_PER_GPU; i++)); do
        SLOT_GPU+=("$g")
    done
done
NUM_SLOTS=${#SLOT_GPU[@]}
declare -a SLOT_PID
for ((s=0; s<NUM_SLOTS; s++)); do SLOT_PID[$s]=""; done

launch_task() {
    local slot_idx="$1"
    local task="$2"
    IFS='|' read -r algo cfg seed <<< "$task"

    local gpu="${SLOT_GPU[$slot_idx]}"
    local cfg_base
    cfg_base="$(basename "$cfg" .yaml)"
    local tag="${algo}__${cfg_base}__seed${seed}"
    local logfile="$LOG_ROOT/${tag}.log"

    local script project
    if [[ "$algo" == "dqn" ]]; then
        script="$RUN_DQN"; project="$DQN_PROJECT"
    else
        script="$RUN_SAC"; project="$SAC_PROJECT"
    fi

    local wandb_args=( --use_wandb --wandb_project "$project" --wandb_mode "$WANDB_MODE" )
    if [[ -n "$WANDB_ENTITY" ]]; then
        wandb_args+=( --wandb_entity "$WANDB_ENTITY" )
    fi

    echo "[slot $slot_idx | gpu $gpu] -> $tag"

    if [[ "$DRY_RUN" == "1" ]]; then
        SLOT_PID[$slot_idx]=""
        return
    fi

    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        # After masking, the only visible GPU is index 0.
        cd "$HW3_DIR"
        python "$script" \
            --config_file "$cfg" \
            --seed "$seed" \
            --which_gpu 0 \
            "${wandb_args[@]}" \
            > "$logfile" 2>&1
    ) &
    SLOT_PID[$slot_idx]=$!
}

# ---------- Dispatch loop ----------
NEXT_TASK=0
NUM_TASKS=${#TASKS[@]}

if [[ "$DRY_RUN" == "1" ]]; then
    for ((t=0; t<NUM_TASKS; t++)); do
        slot_idx=$(( t % NUM_SLOTS ))
        launch_task "$slot_idx" "${TASKS[$t]}"
    done
    echo "DRY_RUN: printed ${NUM_TASKS} task assignments and exiting."
    exit 0
fi

while (( NEXT_TASK < NUM_TASKS )); do
    # Fill empty slots.
    for ((s=0; s<NUM_SLOTS; s++)); do
        if (( NEXT_TASK >= NUM_TASKS )); then break; fi
        pid="${SLOT_PID[$s]}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            launch_task "$s" "${TASKS[$NEXT_TASK]}"
            NEXT_TASK=$(( NEXT_TASK + 1 ))
        fi
    done
    sleep 5
done

echo "All tasks launched. Waiting for in-flight runs to finish..."
wait
echo "Done. Logs: $LOG_ROOT"
