#! /bin/bash
# ============================================================
# 批量训练脚本 - ActivityNet 超参数搜索
# 用法: bash script/train_activitynet_experiments.sh
# ============================================================
#
# 实验设计逻辑：
# - exp01: 降低 alpha_pull (5→1)，降低 alpha_inter_push (1→0.5)
# - exp02: 中间路线，介于 config.json 和 config_refact 之间
# - exp03: 保持 config.json 损失权重，降低学习率 (4e-4→2e-4)
# - exp04: 保持 config.json 损失权重，增加 warmup (400→800)
# - exp05: 反其道行之，增大 push 力度 (push/pull 比 > config.json)
# - exp06: 接近 config_refact，但 alpha_pull 稍大 (0.2→0.5)

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Experiment 01: reduce pull, reduce push"
echo "alpha_pull=1, alpha_inter_push=0.5"
echo "=========================================="
python train.py --config-path config/activitynet/config_exp01.json --seed 8

echo "=========================================="
echo "Experiment 02: middle ground"
echo "alpha_pull=2, alpha_inter_push=0.7, alpha_ivc=2"
echo "=========================================="
python train.py --config-path config/activitynet/config_exp02.json --seed 8

echo "=========================================="
echo "Experiment 03: lower learning rate"
echo "lr=2e-4, config.json loss weights"
echo "=========================================="
python train.py --config-path config/activitynet/config_exp03.json --seed 8

echo "=========================================="
echo "Experiment 04: longer warmup"
echo "warmup=800, config.json loss weights"
echo "=========================================="
python train.py --config-path config/activitynet/config_exp04.json --seed 8

echo "=========================================="
echo "Experiment 05: stronger push"
echo "alpha_pull=3, alpha_inter_push=1.5 (push-heavy)"
echo "=========================================="
python train.py --config-path config/activitynet/config_exp05.json --seed 8

echo "=========================================="
echo "Experiment 06: near refact, slightly stronger pull"
echo "alpha_pull=0.5, alpha_inter_push=0.3"
echo "=========================================="
python train.py --config-path config/activitynet/config_exp06.json --seed 8

echo "All experiments complete!"
