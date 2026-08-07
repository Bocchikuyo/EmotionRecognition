from ultralytics import YOLO


def train():
    # load a model
    model = YOLO("model/yolo26n-cls.pt")

    # train the model
    model.train(data=r"D:\code\PythonProject\EmotionRecognition\data\emotion-data", epochs=100)


if __name__ == '__main__':
    train()
