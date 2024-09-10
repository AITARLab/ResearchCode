# -*- coding = utf-8 -*-
# @time:2024/7/21 15:32
# Author:lyh
# @File:train.py
# @Software:PyCharm
if __name__ == '__main__':
    import os
    import shutil
    from ultralytics import YOLO


    def merge_datasets(train_folds, val_fold, temp_dir):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'images/train'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'images/val'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'labels/train'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'labels/val'), exist_ok=True)

        # 合并训练数据
        for fold in train_folds:
            for subset in ['images', 'labels']:
                src_dir = os.path.join(fold, subset)
                dst_dir = os.path.join(temp_dir, os.path.join(subset, 'train'))
                for file_name in os.listdir(src_dir):
                    shutil.copy(os.path.join(src_dir, file_name), dst_dir)

        # 合并验证数据
        for subset in ['images', 'labels']:
            src_dir = os.path.join(val_fold, subset)
            dst_dir = os.path.join(temp_dir, os.path.join(subset, 'val'))
            for file_name in os.listdir(src_dir):
                shutil.copy(os.path.join(src_dir, file_name), dst_dir)


    folds = [f'dataset/train/fold{i}' for i in range(5)]
    model_path = 'yolov8s.pt'
    temp_dir = 'temp_dataset'
    # 五折交叉验证
    for fold_idx in range(len(folds)):
        print(f'Training on Fold {fold_idx + 1}/{len(folds)}')

        # 创建数据配置文件
        train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
        val_fold = folds[fold_idx]

        # 合并数据集
        merge_datasets(train_folds, val_fold, temp_dir)

        # 加载模型
        model = YOLO(model_path)
        # 训练模型
        results = model.train(
            data='data.yaml',
            epochs=50,
            imgsz=512,
            batch=16,
            name=f'custom_yolo_fold_{fold_idx}',
            device='cuda:0',
            optimizer='Adam',
            lr0=0.0005,
            momentum=0.937,
            weight_decay=0.0005,
        )
