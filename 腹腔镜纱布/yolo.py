from ultralytics import YOLO
import os
from data_process.preprocess import save_txt

def get_yolo_data(file):
    save_file = "yaml/yolo_{}".format(file.split("/")[-1].split("\\")[-1])
    new_data = []
    with open(file,"r") as f:
        data = f.read().split("\n")
        for i in data:
            new_data.append(f"data/images/{i}")
    save_txt(new_data, save_file)

def yolo_train(fold):
    weights_path = "yolov8n.pt"
    model = YOLO(weights_path, task='detect')
    for f in range(fold):
        data_yaml = "yaml/{}fold.yaml".format(f)
        model.train(data=data_yaml, epochs=100, optimizer="Adam",
                    lr0=5e-4, batch=4, device=1, seed=17, val=True, save=True)

def yolo_eval(pth):
    model = YOLO(pth, task='detect')
    results = model.val(data="yaml/4fold.yaml",batch=1,iou=0.4,plots=True,save_json=True, save=True, split="test",device=1)
    # print(results)