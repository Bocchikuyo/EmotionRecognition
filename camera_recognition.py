from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
TRAINED_MODEL_PATH = ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"
BASE_MODEL_PATH = ROOT / "model" / "yolo26n-cls.pt"


def get_model_path():
    return TRAINED_MODEL_PATH if TRAINED_MODEL_PATH.exists() else BASE_MODEL_PATH


def camera_recognition():
    """调用摄像头进行实时情绪识别，按 ESC 退出。"""
    model = YOLO(str(get_model_path()))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查设备或索引。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(frame)[0]
            names_dict = result.names
            probs = result.probs.data.tolist()
            label = names_dict[np.argmax(probs)]

            cv2.putText(
                frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_recognition()
