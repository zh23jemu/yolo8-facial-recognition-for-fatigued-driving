#!/bin/bash
#
# Slurm GPU 节点 YOLOv8 短跑测试脚本。
# 作用：只训练 3 个 epoch，用来验证账户、分区、GPU、依赖、数据路径和代码入口是否全部可用。
# 通过该脚本后，再提交 train_yolo.sbatch 跑正式训练。
#
# 提交命令：
#   sbatch scripts/slurm/train_yolo_test.sbatch
#
# 查看日志：
#   tail -f slurm_logs/fatigue-yolo-test-作业ID.out

#SBATCH --job-name=fatigue-yolo-test
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# 限制常见数值库线程数，避免在 Slurm CPU 绑定环境里创建过多线程。
# 这些设置不会影响 GPU 训练本身，但能减少 ONNX/NumPy/OpenMP 的线程亲和性警告。
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# 切换到提交作业时所在目录，确保 configs、data、src 等相对路径正确。
cd "${SLURM_SUBMIT_DIR}"

echo "========== Slurm Short Test Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Submit Dir: ${SLURM_SUBMIT_DIR}"
echo "Start Time: $(date)"
echo "==========================================="

# 如果服务器要求加载 CUDA/Python 模块，请在这里按实际环境取消注释并修改版本。
# module purge
# module load cuda/12.1
# module load python/3.11

# 项目规则：始终使用项目本地虚拟环境，不直接使用系统 Python 运行训练。
PYTHON_BIN=".venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "未找到 ${PYTHON_BIN}。"
  echo "请先在项目根目录创建并安装依赖："
  echo "  python3.11 -m venv .venv"
  echo "  .venv/bin/python -m pip install --upgrade pip"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

# 输出 GPU 与 PyTorch CUDA 状态，便于确认作业确实跑在 GPU 节点上。
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

# 短跑训练：只跑 3 轮、小 batch、独立实验名，避免覆盖正式训练结果。
# --project 使用绝对路径，避免 Ultralytics 服务器全局 settings 把结果嵌套到 runs/detect/runs/yolo。
"${PYTHON_BIN}" -m src.train.train_yolo \
  --data configs/yolo_data.yaml \
  --model yolov8n.pt \
  --epochs 3 \
  --batch 8 \
  --workers 4 \
  --imgsz 640 \
  --device 0 \
  --project "${SLURM_SUBMIT_DIR}/runs/yolo" \
  --name fatigue_yolov8n_slurm_test

echo "Short test finished at: $(date)"
