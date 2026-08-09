from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emotion-data"


def train():
    model = YOLO("yolo26s_emotion.yaml").load("/model/yolo26s-cls.pt") # 加载 s 版本的预训练权重

    model.train(
        data=str(DATA_PATH),
        epochs=100,
        project="SPD_EMA_CA_s",
        name="train",
        exist_ok=True,  # 如果项目已存在，是否覆盖

        # --- 防过拟合关键配置 ---
        patience=20,  # 如果 20 轮 val 没提升就停止
        weight_decay=0.002,  # 增大权重衰减 (默认 0.0005)
        dropout=0.2,  # 开启 dropout (丢弃 20% 神经元)
        cos_lr=True,  # 开启余弦学习率
        erasing=0.4,  # 开启随机擦除数据增强
        label_smoothing=0.1,  # 缓解情绪分类的主观模棱两可性
        freeze=3,  # 冻结底层通用特征（3层，不能包含自定义模块），专注微调深层情绪特征，
        # auto_augment='randaugment'  # 开启自动数据增强 (分类任务常用)
    )


if __name__ == "__main__":
    train()
