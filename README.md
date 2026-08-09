# EmotionRecognition

本项目最初基于 `YOLO26n-cls` 做情绪分类，完成了训练、单图预测和本地摄像头实时识别等基础功能，初始模型准确率为 `0.847`。

## 原始模型

- 基线模型：`yolo26n-cls.pt`
- 任务：4 类情绪分类
- 数据集：`data/emotion-data/`
- 基础流程：训练、预测、摄像头实时识别

## 优化思路

- 引入 `EMA / CA` 注意力机制，增强特征感知
- 使用 `SPD-Conv`，减少微表情信息丢失
- 设置 `freeze=3`，冻结 Backbone 前 3 层，聚焦深层注意力模块与分类头
- 使用 `patience=20`、`weight_decay=0.002`、`dropout=0.2`、`cos_lr=True`、`erasing=0.4` 抑制过拟合
- 将预训练权重从 `yolo26n` 升级为 `yolo26s`

## 优化结果

- 准确率从 `0.847` 提升到 `0.853`
- 模型对微表情和细粒度情绪特征的感知能力更强

## 核心脚本

- `train.py`：训练脚本，基于 `yolo26s_emotion.yaml` 和 `model/yolo26s-cls.pt` 训练新模型
- `predict.py`：单图预测脚本，优先加载训练好的 `best.pt`
- `camera_recognition.py`：调用本地摄像头进行实时情绪识别

## 文件说明

- `yolo26s_emotion.yaml`：优化后的模型结构配置
- `model/`：预训练权重目录
- `data/`：训练与复现所需数据集
- `runs/classify/SPD_EMA_CA_s/train/weights/best.pt`：训练完成后的可直接使用模型

## 使用方法

```bash
pip install -r requirements.txt
python train.py
python predict.py
python camera_recognition.py
```

## 备注

- `runs` 目录下仅保留训练好的 `best.pt`
- `data/` 可供复现者直接使用，也可用于后续再训练和优化
