#!/bin/bash
#
# Slurm GPU 节点 YOLOv8 训练脚本。
# 使用前请根据服务器实际情况修改 --partition、--account、--gres、module load 等配置。
#
# 提交命令示例：
#   sbatch scripts/slurm/train_yolo.sbatch
#
# 查看队列：
#   squeue -u $USER

#SBATCH --job-name=fatigue-yolov8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# 切换到提交 sbatch 时所在的项目目录。
# 这样可以保证 configs、data、src 等相对路径都从仓库根目录解析。
cd "${SLURM_SUBMIT_DIR}"

echo "========== Slurm Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Submit Dir: ${SLURM_SUBMIT_DIR}"
echo "Start Time: $(date)"
echo "===================================="

# 如果服务器使用 module 管理 CUDA/Python，请在这里按实际环境加载。
# 示例：
# module purge
# module load cuda/12.1
# module load python/3.11

# 项目要求始终使用本地 .venv，不依赖系统 Python 直接运行训练。
PYTHON_BIN=".venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "未找到 ${PYTHON_BIN}，请先在登录节点创建虚拟环境并安装依赖。"
  echo "示例：python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

# 训练前输出 GPU 与 Python 环境，便于论文实验记录和排查问题。
nvidia-smi || true
"${PYTHON_BIN}" - <<'PY'
import sys
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("错误：PyTorch 未能启用 CUDA。请检查服务器虚拟环境中的 torch CUDA 版本。")
    print("建议重新安装：.venv/bin/python -m pip install -r requirements-server-cu121.txt")
    sys.exit(1)
PY

# 正式训练命令。
# 注意：--device 0 表示使用当前 Slurm 分配给作业的第一张 GPU。
"${PYTHON_BIN}" -m src.train.train_yolo \
  --data configs/yolo_data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --project runs/yolo \
  --name fatigue_yolov8n_slurm \
  --export-onnx

echo "End Time: $(date)"
