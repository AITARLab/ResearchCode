import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import warnings
import torch
from torchvision import transforms

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 调用cuda，在这里默认的是只有一个
trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([256, 256]),
                            transforms.Normalize(mean=0.5, std=0.5)])


class clinical_data:
    def __init__(self):
        excel = pd.read_excel('./Clinical_Features/Clinical_Data_Processed.xlsx')
        self.feature = {}
        header = excel.columns.tolist()
        header = header[3:]
        self.len = 0
        for index, row in excel.iterrows():
            self.len += 1
            patient_id = row['patients']
            datas = []
            for name in header:
                datas.append(float(row[name]))
            self.feature[patient_id] = np.array(datas)

    def get_data(self, patient_id):
        if self.feature.get(patient_id) is None:
            print("CLI: get{} Error".format(patient_id))
            exit(0)
        return self.feature[patient_id]


class image_feature:
    def __init__(self, task_type):
        if task_type == 'PDL1':
            excel = pd.read_csv('./Radiomics_Features/feature_pdl1.csv')
        elif task_type == 'VEGF':
            excel = pd.read_csv('./Radiomics_Features/feature_vegf.csv')
        else:
            excel = None
            print("Data type ERROR-radi:{}".format(task_type))
            exit(0)
        self.feature = {}
        header = excel.columns.tolist()
        header = header[2:]
        for index, row in excel.iterrows():
            patient_id = row['patients']
            datas = []
            for name in header:
                datas.append(float(row[name]))
            self.feature[patient_id] = np.array(datas)

    def get_data(self, patient_id):
        if self.feature.get(patient_id) is None:
            print("RAD: get{} Error".format(patient_id))
            exit(0)
        return self.feature[patient_id]


class image_data:
    def __init__(self, task_type):
        self.type0_list = []
        self.type1_list = []
        self.data_dir = ""
        if task_type == 'PDL1':
            self.data_dir = "./image_data/PD_L1_data"
            file_list = os.listdir('./image_data/PD_L1_data/0/DWI/image')
            for file_name in file_list:
                self.type0_list.append(file_name)
            file_list = os.listdir('./image_data/PD_L1_data/1/DWI/image')
            for file_name in file_list:
                self.type1_list.append(file_name)
        elif task_type == 'VEGF':
            self.data_dir = "./image_data/VEGF_data"
            file_list = os.listdir('./image_data/VEGF_data/0/DWI/image')
            for file_name in file_list:
                self.type0_list.append(file_name)
            file_list = os.listdir('./image_data/VEGF_data/1/DWI/image')
            for file_name in file_list:
                self.type1_list.append(file_name)

    def get_data(self, patient_id):
        DWI_list = []
        TW1_list = []
        TW2_list = []
        label = -1
        if patient_id in self.type0_list:
            label = 0
            im_dir = "{}/0/DWI/image/{}".format(self.data_dir, patient_id)
            if os.path.exists(im_dir):
                image_list = os.listdir(im_dir)
                for image_name in image_list:
                    im_d = np.load("{}/{}".format(im_dir, image_name))
                    mask = np.load("{}/0/DWI/mask/{}/{}".format(self.data_dir, patient_id, image_name))
                    im_d = trans(im_d)
                    mask = trans(mask)
                    mask[mask > 0] = 1
                    im = im_d * mask
                    # im = trans(im_d)
                    # im = im.squeeze(0)
                    im = im.to(device)
                    DWI_list.append(im)
            im_dir = "{}/0/T1WI/image/{}".format(self.data_dir, patient_id)
            if os.path.exists(im_dir):
                image_list = os.listdir(im_dir)
                for image_name in image_list:
                    im_d = np.load("{}/{}".format(im_dir, image_name))
                    mask = np.load("{}/0/T1WI/mask/{}/{}".format(self.data_dir, patient_id, image_name))
                    im_d = trans(im_d)
                    mask = trans(mask)
                    mask[mask > 0] = 1
                    im = im_d * mask
                    # im = trans(im_d)
                    # im = im.squeeze(0)
                    im = im.to(device)
                    TW1_list.append(im)
            im_dir = "{}/0/T2WI/image/{}".format(self.data_dir, patient_id)
            if os.path.exists(im_dir):
                image_list = os.listdir(im_dir)
                for image_name in image_list:
                    im_d = np.load("{}/{}".format(im_dir, image_name))
                    mask = np.load("{}/0/T2WI/mask/{}/{}".format(self.data_dir, patient_id, image_name))
                    im_d = trans(im_d)
                    mask = trans(mask)
                    mask[mask > 0] = 1
                    im = im_d * mask
                    # im = trans(im_d)
                    # im = im.squeeze(0)
                    im = im.to(device)
                    TW2_list.append(im)
        elif patient_id in self.type1_list:
            label = 1
            im_dir = "{}/1/DWI/image/{}".format(self.data_dir, patient_id)
            if os.path.exists(im_dir):
                image_list = os.listdir(im_dir)
                for image_name in image_list:
                    im_d = np.load("{}/{}".format(im_dir, image_name))
                    mask = np.load("{}/1/DWI/mask/{}/{}".format(self.data_dir, patient_id, image_name))
                    im_d = trans(im_d)
                    mask = trans(mask)
                    mask[mask > 0] = 1
                    im = im_d * mask
                    # im = trans(im_d)
                    # im = im.squeeze(0)
                    im = im.to(device)
                    DWI_list.append(im)
            im_dir = "{}/1/T1WI/image/{}".format(self.data_dir, patient_id)
            if os.path.exists(im_dir):
                image_list = os.listdir(im_dir)
                for image_name in image_list:
                    im_d = np.load("{}/{}".format(im_dir, image_name))
                    mask = np.load("{}/1/T1WI/mask/{}/{}".format(self.data_dir, patient_id, image_name))
                    im_d = trans(im_d)
                    mask = trans(mask)
                    mask[mask > 0] = 1
                    im = im_d * mask
                    # im = trans(im_d)
                    # im = im.squeeze(0)
                    im = im.to(device)
                    TW1_list.append(im)
            im_dir = "{}/1/T2WI/image/{}".format(self.data_dir, patient_id)
            if os.path.exists(im_dir):
                image_list = os.listdir(im_dir)
                for image_name in image_list:
                    im_d = np.load("{}/{}".format(im_dir, image_name))
                    mask = np.load("{}/1/T2WI/mask/{}/{}".format(self.data_dir, patient_id, image_name))
                    im_d = trans(im_d)
                    mask = trans(mask)
                    mask[mask > 0] = 1
                    im = im_d * mask
                    # im = trans(im_d)
                    # im = im.squeeze(0)
                    im = im.to(device)
                    TW2_list.append(im)
        else:
            label = 2

        if label == -1 or label == 2:
            print("Error")
            exit(0)
        return DWI_list, TW1_list, TW2_list, label


class all_dataset:
    def __init__(self, task_type):
        self.d1 = clinical_data()
        self.d2 = image_feature(task_type)
        self.d3 = image_data(task_type)

    def __len__(self):
        return self.d1.len

    def get_data(self, patient_id):
        f1 = self.d1.get_data(patient_id)
        f2 = self.d2.get_data(patient_id)
        DWI, TW1, TW2, label = self.d3.get_data(patient_id)
        return f1, f2, DWI, TW1, TW2, label


class CCA_dataset(Dataset):
    def __init__(self, task_type, data_txt):
        self.now_read = all_dataset(task_type)
        self.p_id = []
        id_read = open(data_txt, 'r')
        for i in id_read:
            i = i.strip()
            self.p_id.append(i)

    def __len__(self):
        return len(self.p_id)

    def __getitem__(self, idx):
        return self.now_read.get_data(self.p_id[idx])


# if __name__ == "__main__":
#     test_1 = image_data('PDL1')
#     a, b, c, d = test_1.get_data('P1')
#     print(len(a), len(b), len(c), d, a[0].shape)
#     test_1 = clinical_data()
#     test_2 = image_feature('PDL1')
#     test_3 = image_feature('VEGF')
#     a = test_1.get_data('P1')
#     b = test_2.get_data('P2')
#     c = test_3.get_data('P3')
#     print(len(a), len(b), len(c))

