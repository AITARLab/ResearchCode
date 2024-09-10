import json, os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from inference import seg_inference

def draw_gt_bounding_boxes(image_path, json_path, output_path):
    # print(1)
    # 读取图像
    image = Image.open(image_path)
    # plt.figure(figsize=(image.width / 100, image.height / 100), dpi=100)
    plt.imshow(image)
    
    # 读取 JSON 数据
    with open(json_path, 'r',encoding="utf-8") as f:
        data = json.load(f)

    shapes = data["shapes"]

    for shape in shapes:
        shape_type = shape["shape_type"]
        points = shape["points"]
        # label = f"object {shape['group_id']}"
        label = shape["label"]
        # print(label)
        if shape_type == "rectangle":
            x_min, y_max = min(points[0][0], points[1][0]), max(points[0][1],points[1][1])
            x_max, y_min = max(points[0][0], points[1][0]), min(points[0][1],points[1][1])
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='blue', fill=False, linewidth=2))
            # plt.text(x_min, y_min, label, color='red', fontsize=12, verticalalignment='bottom')  # 在矩形旁边添加文本
        
        elif 'rupture' in label:
            # 绘制不规则边缘框
            polygon = plt.Polygon(points, edgecolor='green', fill=False, linewidth=2)
            plt.gca().add_patch(polygon)
    
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)  # 保存图像
    plt.close()  # 关闭图像


def select(result_json,thresshold=0.9):
    with open(result_json,"r") as f:
        content = json.load(f)
    print(f"{'-' * 20} {thresshold} {'-' * 20}")
    for data in content:
        if data["score"] >= thresshold:
            print(data["image_id"])
    print(f"{'-' * 50}")

def draw_pred_box(img_id,result_json,output_path,thresshold=0.9):
    with open(result_json,"r") as f:
        content = json.load(f)
    
    image_path = f"data/images/{img_id}.png"
    image = Image.open(image_path)
    plt.imshow(image)

    for data in content:
        if data["score"] >= thresshold and data["image_id"] == img_id:
            points = data["bbox"]
            label = "{:.4f}".format(data["score"])
            
            plt.gca().add_patch(plt.Rectangle((points[0],points[1]), points[2], points[3], edgecolor='blue', fill=False, linewidth=2))
            plt.text(points[0], points[1], label, color='red', fontsize=12, verticalalignment='bottom') 

    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)  # 保存图像
    plt.close()  # 关闭图像

if __name__ == "__main__":
    image_path = 'data/predict_liver/P4_3.PNG'  # 原始图像路径
    # json_path = 'data/json/P13_4.json'  # JSON 文件路径
    output_path = 'results/fig3_pred.png'  # 输出图像路径
    # select("runs/detect/val2/predictions.json")
    # draw_gt_bounding_boxes(image_path, json_path, output_path)
    # draw_pred_box(img_id="V16_16",result_json="runs/detect/val3/predictions.json",output_path=output_path)
    # pth="weights/fcn_resnet101_4fold_1epoch_0.9722ACC_0.9558IOU_0.9770DICE.pth"
    pth = "weights/fcn_resnet50_0fold_86epoch_0.8621ACC_0.7574IOU_0.8447DICE.pth"
    seg_inference(model_path=pth, img_path=image_path,save_path=output_path)
 