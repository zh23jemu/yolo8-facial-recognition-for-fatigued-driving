#!/bin/bash
#
# YOLOv8n + CBAM 注意力模型正式训练脚本。
# 训练完成后可与 runs/yolo/fatigue_yolov8n_slurm 中的 YOLOv8n 基线结果对比。

#SBATCH --job-name=fatigue-yolov8-cbam
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${SLURM_SUBMIT_DIR}"

echo "========== YOLOv8n-CBAM Slurm Job =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Submit Dir: ${SLURM_SUBMIT_DIR}"
echo "Start Time: $(date)"
echo "============================================"

PYTHON_BIN=".venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "未找到 ${PYTHON_BIN}，请先创建虚拟环境并安装依赖。"
  exit 1
fi

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
    print("错误：PyTorch 未能启用 CUDA。")
    sys.exit(1)
PY

"${PYTHON_BIN}" -m src.train.train_yolo \
  --data configs/yolo_data.yaml \
  --model configs/yolov8n_cbam.yaml \
  --pretrained-weights yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --workers 4 \
  --imgsz 640 \
  --device 0 \
  --project "${SLURM_SUBMIT_DIR}/runs/yolo" \
  --name fatigue_yolov8n_cbam \
  --export-onnx

echo "End Time: $(date)"

