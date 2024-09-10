import nibabel as nib
import pydicom
from matplotlib import pyplot as plt
import os
import numpy as np

print("P23".split('P'))
# 加载NII.GZ文件
mask_file = 'DeepLearning_Data/PD_L1/Label(+)/Mask/P1/1-DWI.nii.gz'
mask_img = nib.load(mask_file)
mask_data = mask_img.get_fdata()
print(mask_data.shape[2])
#
# # 加载DICOM文件夹
# dicom_files = 'DeepLearning_Data/PD_L1/Label(+)/Original_Images/P1/DWI'
# ds = pydicom.dcmread("DeepLearning_Data/PD_L1/Label(+)/Original_Images/P1/DWI/1.3.46.670589.11.34270.5.0.6604.2017120416012814057.156.dcm")
#
# # 获取DICOM图像尺寸
# rows = ds.Rows
# cols = ds.Columns
# slices = len(os.listdir(dicom_files))
# img_array = np.zeros((slices, rows, cols), dtype=ds.pixel_array.dtype)
#
# print(rows, cols)

# 遍历DICOM文件并加载数据
# for i, s in enumerate(sorted(os.listdir(dicom_files))):
#     ds = pydicom.dcmread(f'{dicom_files}/{s}')
#     img_array[i] = ds.pixel_array



