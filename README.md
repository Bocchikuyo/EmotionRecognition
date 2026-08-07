# EmotionRecognition

基于 `ultralytics` YOLO 分类模型的情绪识别项目，支持训练、单图预测和本地摄像头实时识别。

## 功能

- 使用 `model/yolo26n-cls.pt` 作为初始模型训练
- 对单张图片进行情绪分类预测
- 调用本地摄像头进行实时情绪识别

## 情绪类别

- `angry`
- `happy`
- `sad`
- `surprise`

## 环境依赖

- Python 3.10+
- `numpy`
- `opencv-python`
- `ultralytics`

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

- `train.py` 训练脚本
- `predict.py` 单图预测脚本
- `camera_recognition.py` 摄像头实时识别脚本
- `model/yolo26n-cls.pt` 初始权重

## 使用方法

### 1. 训练

```bash
python train.py
```

训练数据默认读取 `data/emotion-data/`。训练结果会保存在 `runs/classify/train/`。

### 2. 单图预测

```bash
python predict.py
```

优先使用 `runs/classify/train/weights/best.pt`，如果不存在则回退到 `model/yolo26n-cls.pt`。

### 3. 摄像头实时识别

```bash
python camera_recognition.py
```

默认打开 `0` 号摄像头，按 `ESC` 退出。

## 说明

- 路径已改为相对项目根目录，换机器后无需修改绝对路径
- `runs/`、数据缓存和 IDE 文件已加入忽略列表
- 如果你要重新训练，请确保本地存在 `data/emotion-data/` 数据集
