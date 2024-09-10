from train import *
from dataset import CCA_dataset
from model import multi_model
import torch
from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 调用cuda，在这里默认的是只有一个
learning_rate = 1e-3
batch_size = 1
num_worker = 0
epochs = 51
patient = 10


def main_solver(task_type):
    test_data = CCA_dataset(task_type, "./{}_test.txt".format(task_type))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_worker, pin_memory=False, shuffle=False)
    AUC_vali_list = []
    AUC_test_list = []
    for fold in range(4):
        print("Training fold:{}".format(fold))
        train_data = CCA_dataset(task_type, "./{}_train_fold{}.txt".format(task_type, fold))
        val_data = CCA_dataset(task_type, "./{}_val_fold{}.txt".format(task_type, fold))
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_worker, pin_memory=False,
                                  shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_worker, pin_memory=False, shuffle=True)
        now_model = multi_model(2)
        now_model = now_model.to(device)
        patient_cnt = 0
        best_AUC = 0
        best_test_AUC = 0
        optimizer = torch.optim.Adam(now_model.parameters(), lr=learning_rate, weight_decay=1e-5)
        for epoch in range(1, epochs):
            new_AUC = train(now_model, train_loader, val_loader, device, epoch, fold, optimizer, task_type, best_AUC)
            if new_AUC > best_AUC:
                patient_cnt = 0
                print("Update vali AUC: {:.4f}->{:.4f}".format(best_AUC, new_AUC))
                best_AUC = new_AUC
            new_AUC = experiment(now_model, test_loader, device, fold, task_type, best_test_AUC)
            if new_AUC > best_test_AUC:
                patient_cnt = 0
                print("Update test AUC: {:.4f}->{:.4f}".format(best_test_AUC, new_AUC))
                best_test_AUC = new_AUC
            patient_cnt += 1
            if patient_cnt > patient:
                print("Patient down! stop training.")
                break
            if epoch % 10 == 0:
                print("Epoch-{}/{}: vali-AUC:{:.4f} test-AUC:{:.4f}".format(epoch, epochs-1, best_AUC, best_test_AUC))
        AUC_vali_list.append(best_AUC)
        AUC_test_list.append(best_test_AUC)
    AUC_vali_list = np.array(AUC_vali_list)
    AUC_test_list = np.array(AUC_test_list)
    print("Max vali-AUC:{:.4f}, mean:{:.4f}".format(np.max(AUC_vali_list), np.mean(AUC_vali_list)))
    print("Max test-AUC:{:.4f}, mean:{:.4f}".format(np.max(AUC_test_list), np.mean(AUC_test_list)))


if __name__ == "__main__":
    main_solver("PDL1")
    main_solver("VEGF")
