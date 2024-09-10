from model import multi_model
from metric import *
import torch
import torch.nn as nn
from vis import *


def train(model, train_data, val_data, device, epoch, fold, optimizer, task_type, best_AUC=0):
    model.train()
    cnt = 0
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()
    loss_fn.to(device)
    for f1, f2, DWI, TW1, TW2, label in train_data:
        f1, f2, label = f1.to(device), f2.to(device), label.to(device)
        optimizer.zero_grad()
        output, _ = model(f1, f2, DWI, TW1, TW2)
        # print(output)
        # loss = loss_fn(output, label.float())
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        if cnt % 20 == 0:
            print("{}-Training loss-{}: {:.6f}".format(epoch, cnt, loss))
        cnt += 1
    model.eval()
    prob_result = []
    predict_list = []
    label_true = []
    with torch.no_grad():
        for f1, f2, DWI, TW1, TW2, label in val_data:
            f1, f2, label = f1.to(device), f2.to(device), label.to(device)
            _, output = model(f1, f2, DWI, TW1, TW2)
            output = output.cpu().detach().numpy()
            label_true.append(label[0].cpu().item())
            predict_list.append(np.argmax(output, axis=1))
            prob_result.append(output[0][1])
    vali = Metrics(label_true, predict_list, prob_result)
    ACC = vali.accuracy()
    AUC = vali.AUC()
    F1 = vali.f1()
    precision = vali.precision()
    sensitivity = vali.sensitivity()
    specificity = vali.specificity()
    # print(label_true, prob_result, predict_list)
    lower, upper = bootstrap_auc(label_true, prob_result, [0, 1])
    print("VALI:Fold{}-Epoch{}: ACC:{:.4f}, AUC:{:.4f}, F1:{:.4f}, pre:{:.4f}, sen:{:.4f}, spe:{:.4f} 95%CI:{:.4f}-{:.4f}\n".format(fold, epoch,
                                                                                                           ACC, AUC, F1, precision, sensitivity, specificity, lower, upper))
    if AUC > best_AUC:
        torch.save(model, "./weights/{}/vali-fold{}-AUC{:.4f}-ACC{:.4f}.pth".format(task_type, fold, AUC, ACC))
        draw_roc(label_true, prob_result, "{}-vali".format(fold), "./result/{}".format(task_type))
        draw_cm(label_true, predict_list, "{}-vali".format(fold), [0, 1], "./result/{}".format(task_type))
    return AUC


def experiment(model, test_data, device, fold, task_type, best_AUC=0):
    model.eval()
    prob_result = []
    predict_list = []
    label_true = []
    with torch.no_grad():
        for f1, f2, DWI, TW1, TW2, label in test_data:
            f1, f2, label = f1.to(device), f2.to(device), label.to(device)
            _, output = model(f1, f2, DWI, TW1, TW2)
            output = output.cpu().detach().numpy()
            label_true.append(label.cpu().item())
            predict_list.append(np.argmax(output, axis=1))
            prob_result.append(output[0][1])
    vali = Metrics(label_true, predict_list, prob_result)
    ACC = vali.accuracy()
    AUC = vali.AUC()
    F1 = vali.f1()
    precision = vali.precision()
    sensitivity = vali.sensitivity()
    specificity = vali.specificity()
    lower, upper = bootstrap_auc(label_true, prob_result, [0, 1])
    print("TEST:Fold{}: ACC:{:.4f}, AUC:{:.4f}, F1:{:.4f}, pre:{:.4f}, sen:{:.4f}, spe:{:.4f} 95%CI:{:.4f}-{:.4f}\n".format(fold,
                                                                                                           ACC, AUC, F1, precision, sensitivity, specificity, lower, upper))
    if AUC > best_AUC:
        torch.save(model, "./weights/{}/test-fold{}-AUC{:.4f}-ACC{:.4f}.pth".format(task_type, fold, AUC, ACC))
        draw_roc(label_true, prob_result, "{}-test".format(fold), "./result/{}".format(task_type))
        draw_cm(label_true, predict_list, "{}-test".format(fold), [0, 1], "./result/{}".format(task_type))
    return AUC
