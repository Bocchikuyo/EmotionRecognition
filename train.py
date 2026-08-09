from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emotion-data"
MODEL_YAML = ROOT / "yolo26s_emotion.yaml"
PRETRAINED_WEIGHT = ROOT / "model" / "yolo26s-cls.pt"


def train():
    model = YOLO(str(MODEL_YAML)).load(str(PRETRAINED_WEIGHT))
    model.train(
        data=str(DATA_PATH),
        epochs=100,
        project="SPD_EMA_CA_s_smt",
        name="train",
        exist_ok=True,
        patience=20,
        weight_decay=0.002,
        dropout=0.2,
        cos_lr=True,
        erasing=0.4,
        freeze=3,
    )


if __name__ == "__main__":
    train()
