import os.path

from draw_analysis import *
import torch
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib_inline
from torchvision import transforms
import numpy as np


def GT_mask_draw(img_path, mask_path):
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        mask = np.array(Image.open(mask_path))/255
        mask_array = mask[:, :, np.newaxis]
        img_with_mask = draw_mask(img_array, mask_array, color='green', alpha=0.8)
        save_path = "results/injury_GT.png"
        plt.imsave(save_path, img_with_mask)


def seg_inference(model_path, img_path):
    """

    :param model_path:
    :param img_path:
    :return:
    """
    # 加载模型
    model = torch.load(model_path, map_location='cpu')

    # 输入图像处理
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_array = np.array(img)
    w, h = img.size[0], img.size[1]
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # 预测结果
    pred = model(trans(img_array).unsqueeze(dim=0))
    pred = torch.argmax(pred, dim=1)

    # print(pred.shape)
    pred = transforms.Resize([h, w])(pred)
    # 制图
    pred = pred.squeeze(dim=0)
    pred = pred.detach().numpy()

    indices = np.where(pred == 1)
    min_h_idx = (indices[0].min() - 6) if (indices[0].min() - 6) > 0 else 0
    max_h_idx = (indices[0].max() + 6) if (indices[0].max() + 6) < h else h
    min_w_idx = (indices[1].min() - 6) if (indices[1].min() - 6) > 0 else 0
    max_w_idx = (indices[1].max() + 6) if (indices[1].max() + 6) < w else w
    pred[min_h_idx: max_h_idx, min_w_idx: max_w_idx] = 1
    mask_array = pred[:, :, np.newaxis]
    result = image_bitwise_mask(img, mask_array)
    save_path = "data/predict_spleen/{}".format(img_path.split("\\")[-1])
    result.save(save_path)

    # source_img_path = img_path.replace("predict_liver", "image")
    # source_img = Image.open(source_img_path).convert("RGB")
    # source_img_array = np.array(source_img)
    # img_with_mask = draw_mask(image=source_img_array, mask=mask_array, color='yellow', alpha=0.8)
    # plt.imsave("results/injury_pred.png", img_with_mask)
    # save_path = "data/predict_spleen/{}".format(img_path.split("\\")[-1])
    # result.save(save_path)
    # plt.imsave(save_path, result)
    # plt.imshow(result)
    # plt.show()



