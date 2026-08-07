import numpy as np
from ultralytics import YOLO


def predict():
    model_path = "runs/classify/train/weights/best.pt"
    image_path = r"D:\code\PythonProject\EmotionRecognition\data\emotion-data\val\angry\PrivateTest_1109992.jpg"

    model = YOLO(model_path)
    result = model(image_path)
    # print(result)

    names_dict = result[0].names
    probs = result[0].probs.data.tolist()
    print(names_dict)
    print(probs)

    # 找印分类结果
    print(f'分类结果: {names_dict[np.argmax(probs)]}')


if __name__ == '__main__':
    predict()