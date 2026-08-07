from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "yolo26n-cls.pt"
DATA_PATH = ROOT / "data" / "emotion-data"


def train():
    model = YOLO(str(MODEL_PATH))
    model.train(data=str(DATA_PATH), epochs=100)


if __name__ == "__main__":
    train()
