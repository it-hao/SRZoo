# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: HAOSHEN
"""
import argparse
import torch
import torch.optim as optim
from flyai.dataset import Dataset
from model import Model
from bcan import Net
from path import MODEL_PATH
import torch.nn as nn
import utility

# 这是 https://www.flyai.com/ 中的超分辨率题，评判标准就是测试SSIM的值，图片的数据集官方提供，提供了数据集的路径接口。
"""
 官方的接口中是有BATCH 和 EPOCHS两个超参数，其中BATCH的多少会决定dataset.get_step()的大小，测试集的总量是540，
 1):当BATCH=1,EPOCH=1时，dataset.get_step()=540 * 1 = 540;
 2):当BATCH=1,EPOCH=10时，dataset.get_step()=540 * 10 = 5400;
 3):当BATCH=16,EPOCH=1时，dataset.get_step()=540 // 16 * 1 = 34;
 4):当BATCH=16,EPOCH=10时，dataset.get_step()=540 // 16 * 10 = 340;
 但是这里的batch_size我私自设定了，此处的batch_size是自己定义的，这里的设计为了尽量减少图片的读取时间，
 将每张图片随机切成batch_size个小块，并传入网络中。
 参数设计：
 model: 	BCAN
 LR:		1e-3
 BATCH_SIZE:16
 BATCH:		1
 EPOCHS 	20
"""
parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--BATCH_SIZE", default=16, type=int, help="batch size")
parser.add_argument("-ts", "--TEST_STEP", default=1, type=int, help="frequency of test image")
parser.add_argument("-nt", "--TEST_NUMBER", default=1, type=int, help="number of test image")
parser.add_argument("-lr", "--LR", default=1e-3, type=float, help="initial learning rate")

parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
args = parser.parse_args()

if __name__ == '__main__':
    best_ssim = 0
    best_iter = 1
    dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
    # 读取验证集进行验证，保证验证集一样
    lr_val, hr_val = dataset.next_validation_batch()

    # 参数初始化
    model = Model(dataset)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)
    net = Net().to(device)
    criterion = nn.L1Loss()
    optimize = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), args.LR)
    # ======================开始训练===========================
    net.train()
    iteration = dataset.get_step()  
    print("Batch_size=", args.BATCH_SIZE, "iteration=", iteration, "lr=", args.LR)

    for iter in range(iteration): 
        # 获取 LR/HR 的图片路径
        x_train, y_train = dataset.next_train_batch()

        # 对数据进行处理，这里为了加快速度，每次只读一张图片，然后对一张图片进行batch_size次随机裁剪，得到一个batch
        x_train, y_train = utility.load_image(x_train[0], y_train[0], batch_size = args.BATCH_SIZE, is_train=True)
        # print("---------------------------------")
        # print(x_train.shape, y_train.shape)

        # 数据导入GPU
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        out = net(x_train)
        loss = criterion(out, y_train)
        optimize.zero_grad()
        loss.backward()
        optimize.step()

        learning_rate = args.LR * (0.5 ** (iter // iteration))  # 迭代次数---降低学习率
        for param_group in optimize.param_groups:
            param_group["lr"] = learning_rate
        model.save_model(net, MODEL_PATH, overwrite=True)
        print("====>", str(iter + 1) + "/" + str(iteration), "LR = ", learning_rate, "Loss = ", loss.item())
    # ======================结束训练===========================

    # ======================开始验证===========================
    # total_ssim = 0.0
    # avg_ssim = 0.0
    # if (iter + 1) % args.TEST_STEP == 0:
    #     net.eval()
    #     # 测试五张图片
    #     for i in range(args.TEST_NUMBER):
    #         x_val, y_val = utility.load_image(lr_val[i], hr_val[i], is_train=False)
    #         x_val = torch.unsqueeze(x_val, 0)
    #         y_val = torch.unsqueeze(y_val, 0)
    #         with torch.no_grad():
    #             x_val = x_val.to(device)
    #             y_val = y_val.to(device)
    #             sr_val = net(x_val)
    #             visuals = utility.get_current_visual(y_val, sr_val, rgb_range=255)
    #             img_ssim = utility.calc_metrics(visuals['SR'], visuals['HR'], crop_border=4)
    #             total_ssim += img_ssim
    #     avg_ssim = total_ssim / args.TEST_NUMBER
    #     print("<====>curr iter", iter+1, "==>curr ssim", avg_ssim)
    #     if avg_ssim > best_ssim:
    #         best_iter = iter + 1
    #         best_ssim = avg_ssim
    #         model.save_model(net, MODEL_PATH, overwrite=True)
    #     print("<====>best iter", best_iter, "==>best ssim", best_ssim)
    # ======================结束验证============================



