import cv2
from ultralytics import YOLO
import numpy as np


def camera_recognition():
    """调用摄像头实时识别情绪。按 ESC 退出。"""
    model_path = "runs/classify/train/weights/best.pt"
    image_path = r"D:\code\PythonProject\EmotionRecognition\data\emotion-data\val\angry\PrivateTest_1109992.jpg"

    emotion = ['angry', 'happy', 'sad', 'surprise']

    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('无法打开摄像头，请检查设备或索引。')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = model(frame)
            names_dict = result[0].names
            probs = result[0].probs.data.tolist()
            for i in range(len(emotion)):
                if names_dict[np.argmax(probs)] == emotion[i]:
                    cv2.putText(
                        frame,
                        emotion[i],
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera_recognition()
