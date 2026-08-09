from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emotion-data"


def train():
    model = YOLO("model.yaml")
    model.train(
        data=str(DATA_PATH),
        epochs=100,

        project="SPD_EMA_CA",
        name="train",
        exist_ok=True,  # 如果项目已存在，是否覆盖

        # --- 防过拟合关键配置 ---
        patience=15,  # 如果 15 轮 val 没提升就停止
        weight_decay=0.002,  # 增大权重衰减 (默认 0.0005)
        dropout=0.2,  # 开启 dropout (丢弃 20% 神经元)
        cos_lr=True,  # 开启余弦学习率
        # erasing=0.4,  # 开启随机擦除数据增强
        auto_augment='randaugment'  # 开启自动数据增强 (分类任务常用)
    )


if __name__ == "__main__":
    train()
