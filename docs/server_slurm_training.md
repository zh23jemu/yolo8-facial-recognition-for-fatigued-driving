# Slurm 服务器 GPU 训练说明

本项目正式训练建议放到服务器 GPU 节点完成，本地主要用于代码开发、数据整理和小规模推理验证。

## 1. 上传项目

将项目目录上传到服务器，例如：

```bash
~/projects/yolo8-facial-recognition-for-fatigued-driving
```

大型数据集不提交 Git。请单独上传：

```text
data/fatigue_yolo/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

## 2. 创建服务器虚拟环境

在服务器登录节点进入项目根目录：

```bash
cd ~/projects/yolo8-facial-recognition-for-fatigued-driving
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

如果服务器没有 Python 3.11，可使用服务器提供的 Python 3.10 或管理员推荐版本，但仍需创建项目本地 `.venv`。

## 3. 检查数据配置

训练配置文件：

```text
configs/yolo_data.yaml
```

类别顺序已按 Roboflow 数据集固定为：

```text
0: Eyeclosed
1: Eyeopen
2: Yawn
```

不要随意交换类别顺序，否则会导致闭眼/睁眼标签错位。

## 4. 修改 Slurm 参数

打开：

```text
scripts/slurm/train_yolo.sbatch
```

根据服务器要求修改：

- `#SBATCH --partition=gpu`
- `#SBATCH --gres=gpu:1`
- `#SBATCH --cpus-per-task=8`
- `#SBATCH --mem=32G`
- `#SBATCH --time=12:00:00`
- 如服务器要求项目账号，增加 `#SBATCH --account=你的账号`
- 如服务器要求加载模块，取消 `module load` 示例注释并改成实际版本

## 5. 提交训练

在项目根目录执行：

```bash
sbatch scripts/slurm/train_yolo.sbatch
```

查看队列：

```bash
squeue -u $USER
```

查看训练日志：

```bash
tail -f slurm_logs/fatigue-yolov8-作业ID.out
```

## 6. 训练结果

默认输出目录：

```text
runs/yolo/fatigue_yolov8n_slurm/
```

重点保存：

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `confusion_matrix.png`
- `PR_curve.png`
- 导出的 ONNX 模型

训练完成后，可将最优权重复制到项目约定位置：

```bash
cp runs/yolo/fatigue_yolov8n_slurm/weights/best.pt weights/best.pt
```

## 7. 建议训练策略

第一次提交建议先短跑验证：

```bash
.venv/bin/python -m src.train.train_yolo --data configs/yolo_data.yaml --model yolov8n.pt --epochs 3 --batch 8 --device 0
```

确认数据路径、CUDA 和依赖没有问题后，再用 `sbatch` 跑 50 轮正式训练。

