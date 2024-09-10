import nibabel as nib
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import time
import SimpleITK as s_itk
import cv2


def normalize_img(image):
    """
    normalize image to [0,1]
    :param image: input image, numpy
    :return: normalized image
    """
    _range = image.max() - image.min()
    if _range == 0:
        plt.imshow(image, cmap='gray')
        plt.show()
        assert _range != 0, "image is empty!"
    assert _range > 0
    return (image - image.min()) / _range


def load_scan(path, normalize=True):
    """
    load dcm or other MRI or CT source file to image type
    :param path: the path of MRI source folder
    :param normalize: whether normalize the image, True or False
    :return: an array list of MRI images
    """
    ck_list = os.listdir(path)
    for file_name in ck_list:
        if os.path.isdir("{}/{}".format(path, file_name)):
            return load_scan("{}/{}".format(path, file_name))
        else:
            break
    reader = s_itk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    image_array = s_itk.GetArrayFromImage(image)  # z, y, x
    if normalize:
        norm_image_array = []
        for img in image_array:
            image = normalize_img(img)
            norm_image_array.append(image)
        image_array = norm_image_array
    return image_array


def show_one_batch(source_file_path):
    """
    test for trans one MRI or CT files dir to image
    :param source_file_path: the path of MRI files
    :return: None
    """
    image_array = load_scan(source_file_path)
    for i in image_array:
        plt.imshow(i, cmap='gray')
        plt.show()
        time.sleep(1)


def dcm_to_info(dcm_path, png_path='./out_png', dcm_name=''):
    ds = pydicom.dcmread(dcm_path)
    if len(ds.pixel_array.shape) > 2:
        # 循环提取图片
        for index in range(int(ds.pixel_array.shape[0])):
            # 根据dicom文件的BitAllocation显示16,代表是16位图像
            img = np.asarray(ds.pixel_array[index], dtype='uint8')
            img_upload_path = os.path.join(png_path, "{}-{}.png".format(dcm_name, index))
            # cmap参数是重点，选择plt.cm.bone，胸片显示才会正常
            # plt.axis('off')
            plt.imsave(img_upload_path, img, cmap=plt.cm.bone)
            # , bbox_inches='tight', pad_inches=0)
            # plt.imshow(img, cmap=plt.cm.bone)
    else:
        # 单图
        img = np.asarray(ds.pixel_array, dtype='uint8')
        img_upload_path = os.path.join(png_path, "{}-0.png".format(dcm_name))
        # plt.axis('off')
        plt.imsave(img_upload_path, img, cmap=plt.cm.bone)
        # , bbox_inches='tight', pad_inches=0)
        # plt.imshow(img, cmap=plt.cm.bone)
    # plt.show()


def rotate(image, angle, center=None, scale=1.0):
    """
    rotate image
    :param image: input image, numpy or Image
    :param angle: [0,360] 360 degree
    :param center: rotate center [x, y] int
    :param scale: zoom times (0,+oo) float
    :return: rotated image
    """
    if angle == 0:
        return image.copy()
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h)).copy()
    return rotated


# 翻转 left_right
def flip(image, flip_type='lr'):
    """
    flip image
    :param image: input image, numpy
    :param flip_type: 'lr' for flip horizontal, 'ud' for flip vertical
    :return:
    """
    if flip_type == 'lr':
        flipped_image = np.fliplr(image).copy()
    else:
        flipped_image = np.flipud(image).copy()
    return flipped_image


def read_mask(mask_path, dcm_path, save_path, p_name):
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # if not os.path.exists("{}/mask".format(save_path)):
    #     os.mkdir("{}/mask".format(save_path))
    # if not os.path.exists("{}/image".format(save_path)):
    #     os.mkdir("{}/image".format(save_path))
    mask_obj = nib.load(mask_path)
    mask_data = mask_obj.get_fdata()
    image_array = load_scan(dcm_path)
    cnt = 0
    if mask_data.shape[2] != len(image_array):
        print("Not len fit: {}".format(mask_path))
        return
    for i in range(len(image_array)):
        now_mask = mask_data[:, :, i]
        now_img = image_array[i]
        now_mask = rotate(now_mask, 270)
        # now_mask = flip(now_mask)
        if now_mask.any() > 0:
            # cnt += 1
            # now_mask = np.array(now_mask)
            # now_img = np.array(now_img)
            # if not os.path.exists("{}/mask/{}".format(save_path, p_name)):
            #     os.mkdir("{}/mask/{}".format(save_path, p_name))
            # if not os.path.exists("{}/image/{}".format(save_path, p_name)):
            #     os.mkdir("{}/image/{}".format(save_path, p_name))
            # np.save("{}/mask/{}/{}.npy".format(save_path, p_name, cnt), now_mask)
            # np.save("{}/image/{}/{}.npy".format(save_path, p_name, cnt), now_img)
            # print(now_mask.shape)
            plt.subplot(1, 3, 1)
            plt.imshow(now_img, cmap='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(now_mask, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(now_mask * now_img, cmap='gray')
            plt.show()



def solve(dir_path, save_path, deal=False):
    print(dir_path)
    if not deal:
        label_1 = os.listdir("{}/Label(+)/Mask".format(dir_path))
        dir_1 = "{}/Label(+)".format(dir_path)
        save_path_1 = "{}/1".format(save_path)
        if not os.path.exists(save_path_1):
            os.mkdir(save_path_1)
        print("Deal 1:")
        for p_name in label_1:
            index = int(p_name.split('P')[1])
            print(index)
            if os.path.exists("{}/Mask/{}/{}-DWI.nii.gz".format(dir_1, p_name, index)):
                read_mask("{}/Mask/{}/{}-DWI.nii.gz".format(dir_1, p_name, index),
                          "{}/Original_Images/{}/DWI".format(dir_1, p_name), "{}/DWI".format(save_path_1), p_name)
            else:
                read_mask("{}/Mask/{}/DWI.nii.gz".format(dir_1, p_name),
                          "{}/Original_Images/{}/DWI".format(dir_1, p_name), "{}/DWI".format(save_path_1), p_name)
            if os.path.exists("{}/Mask/{}/{}-T1WI.nii.gz".format(dir_1, p_name, index)):
                read_mask("{}/Mask/{}/{}-T1WI.nii.gz".format(dir_1, p_name, index),
                          "{}/Original_Images/{}/T1WI".format(dir_1, p_name), "{}/T1WI".format(save_path_1), p_name)
            else:
                read_mask("{}/Mask/{}/T1WI.nii.gz".format(dir_1, p_name),
                          "{}/Original_Images/{}/T1WI".format(dir_1, p_name), "{}/T1WI".format(save_path_1), p_name)
            if os.path.exists("{}/Mask/{}/{}-T2WI.nii.gz".format(dir_1, p_name, index)):
                read_mask("{}/Mask/{}/{}-T2WI.nii.gz".format(dir_1, p_name, index),
                          "{}/Original_Images/{}/T2WI".format(dir_1, p_name), "{}/T2WI".format(save_path_1), p_name)
            else:
                read_mask("{}/Mask/{}/T2WI.nii.gz".format(dir_1, p_name),
                          "{}/Original_Images/{}/T2WI".format(dir_1, p_name), "{}/T2WI".format(save_path_1), p_name)
    label_0 = os.listdir("{}/Label(-)/Mask".format(dir_path))
    dir_0 = "{}/Label(-)".format(dir_path)
    save_path_0 = "{}/0".format(save_path)
    if not os.path.exists(save_path_0):
        os.mkdir(save_path_0)
    print("Deal 0:")
    for p_name in label_0:
        index = int(p_name.split('P')[1])
        print(index)
        if os.path.exists("{}/Mask/{}/{}-DWI.nii.gz".format(dir_0, p_name, index)):
            read_mask("{}/Mask/{}/{}-DWI.nii.gz".format(dir_0, p_name, index),
                      "{}/Original_Images/{}/DWI".format(dir_0, p_name), "{}/DWI".format(save_path_0), p_name)
        else:
            read_mask("{}/Mask/{}/DWI.nii.gz".format(dir_0, p_name),
                      "{}/Original_Images/{}/DWI".format(dir_0, p_name), "{}/DWI".format(save_path_0), p_name)
        if os.path.exists("{}/Mask/{}/{}-T1WI.nii.gz".format(dir_0, p_name, index)):
            read_mask("{}/Mask/{}/{}-T1WI.nii.gz".format(dir_0, p_name, index),
                      "{}/Original_Images/{}/T1WI".format(dir_0, p_name), "{}/T1WI".format(save_path_0), p_name)
        else:
            read_mask("{}/Mask/{}/T1WI.nii.gz".format(dir_0, p_name),
                      "{}/Original_Images/{}/T1WI".format(dir_0, p_name), "{}/T1WI".format(save_path_0), p_name)
        if os.path.exists("{}/Mask/{}/{}-T2WI.nii.gz".format(dir_0, p_name, index)):
            read_mask("{}/Mask/{}/{}-T2WI.nii.gz".format(dir_0, p_name, index),
                      "{}/Original_Images/{}/T2WI".format(dir_0, p_name), "{}/T2WI".format(save_path_0), p_name)
        else:
            read_mask("{}/Mask/{}/T2WI.nii.gz".format(dir_0, p_name),
                      "{}/Original_Images/{}/T2WI".format(dir_0, p_name), "{}/T2WI".format(save_path_0), p_name)


if __name__ == "__main__":
    # solve("./DeepLearning_Data/PD_L1", "./DeepLearning_Data/PD_L1_data")
    # solve("./DeepLearning_Data/VEGF", "./DeepLearning_Data/VEGF_data")
    # solve("./DeepLearning_Data/VEGF", "./DeepLearning_Data/VEGF_data", True)
    read_mask("DeepLearning_Data/PD_L1/Label(+)/Mask/P1/1-DWI.nii.gz",
              "DeepLearning_Data/PD_L1/Label(+)/Original_Images/P1/DWI", "", "")
