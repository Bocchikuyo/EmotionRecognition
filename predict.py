from pathlib import Path

import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
TRAINED_MODEL_PATH = ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"
BASE_MODEL_PATH = ROOT / "model" / "yolo26n-cls.pt"
IMAGE_PATH = ROOT / "data" / "emotion-data" / "val" / "angry" / "PrivateTest_1109992.jpg"


def get_model_path():
    return TRAINED_MODEL_PATH if TRAINED_MODEL_PATH.exists() else BASE_MODEL_PATH


def predict():
    model = YOLO(str(get_model_path()))
    result = model(str(IMAGE_PATH))[0]
    names_dict = result.names
    probs = result.probs.data.tolist()
    print(names_dict)
    print(probs)
    print(f"分类结果: {names_dict[np.argmax(probs)]}")

def predict1():
    model = YOLO(str(get_model_path()))
    model.track()
    result = model(str(IMAGE_PATH))[0]
    names_dict = result.names
    probs = result.probs.data.tolist()
    print(names_dict)
    print(probs)
    print(f"分类结果: {names_dict[np.argmax(probs)]}")

if __name__ == "__main__":
    predict()
