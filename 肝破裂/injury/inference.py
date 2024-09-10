from log.draw_analysis import *
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
        save_path = "GT.png"
        plt.imsave(save_path, img_with_mask)



def seg_inference(model_path, img_path, save_path=None):
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
#     plt.imsave(save_path, pred)
    img = Image.open(img_path.replace("predict_liver", "images")).convert("RGB")
    img_array = np.array(img)
    mask_array = pred[:, :, np.newaxis]
#     result = image_bitwise_mask(img, mask_array)
    img_with_mask = draw_mask(image=img_array, mask=mask_array, color='yellow', alpha=0.8)
#     save_path = "result/{}".format(img_path.split("\\")[-1].split("/")[-1])
#     result.save(save_path)
    plt.imsave(save_path, img_with_mask)
#     plt.imshow(result)
#     plt.show()



