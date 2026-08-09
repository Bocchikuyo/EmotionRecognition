# EmotionRecognition

基于 YOLO 分类模型的情绪识别项目。最新版本通过引入 EMA/CA 注意力机制、SPD-Conv、冻结 Backbone 前 3 层以及多项抗过拟合策略，将准确率从 `0.847` 提升到 `0.853`。

## 本次优化

- 引入 `EMA / CA` 增强特征感知
- 使用 `SPD-Conv` 减少微表情信息丢失
- `freeze=3` 冻结 Backbone 前 3 层，聚焦深层注意力模块与分类头
- 使用 `patience=20`、`weight_decay=0.002`、`dropout=0.2`、`cos_lr=True`、`erasing=0.4` 降低过拟合
- 预训练权重由 `yolo26n` 升级为 `yolo26s`

## 效果

- 准确率：`0.847 -> 0.853`
- 更适合微表情与细粒度情绪特征提取

## 文件说明

- `yolo26s_emotion.yaml`：新模型结构配置
- `model/yolo26s-cls.pt`：small 版预训练权重
- `runs/classify/SPD_EMA_CA_s/train/weights/best.pt`：训练好的模型，可直接使用
- `data/`：训练与复现所需数据集

## 使用方法

```bash
pip install -r requirements.txt
python train.py
python predict.py
python camera_recognition.py
```

## 备注

- `runs` 目录下仅保留并上传 `runs/classify/SPD_EMA_CA_s/train/weights/best.pt`
- 其余训练日志与中间结果不提交
