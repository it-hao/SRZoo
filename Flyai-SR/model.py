# -*- coding: utf-8 -*
import numpy as np
import os
import torch
from flyai.model.base import Base
import cv2
import scipy.misc as misc
import skimage.color as sc
from path import MODEL_PATH, LOG_PATH # 日志路径和模型的路径
from bcan import Net
# __import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        self.device = torch.device('cuda')
        if os.path.exists(self.net_path):
            self.net = torch.load(self.net_path)

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)
        outputs = self.net(x_data)
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):  # datas表示的是路径的名字
        labels = []
        self.model = Net().to(self.device)
        self.model.load_state_dict(
            torch.load(self.net_path),
            strict=False
        )
        self.model.eval()
        for data in datas:
            # 图片路径
            with torch.no_grad():
                data = self.data.predict_data(**data)
                data_path = data[0]
                # 加载图片
                lr = load_image(data_path)
                # 扩维
                lr = torch.unsqueeze(lr, 0)
                # 导入gpu
                lr = lr.to(self.device)
                # -------------------------------
                # sr = self.forward_chop(lr)
                # ------------------------------
                # ------------------------------
                forward_chop = self.forward_chop
                sr = self.forward_x8(lr, forward_chop)
                # ------------------------------
                # 将输出的数据转换为numpy
                sr = sr[0].data.permute(1, 2, 0).cpu().numpy()  # 转换维度
                sr = np.clip(sr, 0, 255)
                sr = np.round(sr)  # 转整
                sr = sr.astype(np.uint8)
                sr = self.data.to_categorys(sr)
                labels.append(sr)
                print(data_path, sr.shape)
        return labels
        "----------------------不要动-----------------------------"
        # if self.net is None:
        #     self.net = torch.load(self.net_path)
        #
        # labels = []
        # device = torch.device('cuda')
        #
        # for data in datas:
        #     data = self.data.predict_data(**data)
        #
        #     data_path = data[0]
        #
        #     lr = load_image(data_path)
        #
        #     lr = torch.unsqueeze(lr, 0)
        #
        #     lr = lr.to(device)
        #     sr = self.net(lr)
        #
        #     sr = sr[0].data.permute(1, 2, 0).cpu().numpy()  # 转换维度
        #     sr = np.clip(sr, 0, 255)
        #     sr = np.round(sr)  # 转整
        #     sr = sr.astype(np.uint8)
        #     sr = self.data.to_categorys(sr)
        #     labels.append(sr)
        # return labels
        "----------------------不要动-----------------------------"


    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        # torch.save(network, os.path.join(path, name))
        torch.save(
            network.state_dict(),
            os.path.join(path,  name)
        )

    def forward_chop(self, x, shave=10, min_size=50000):
        scale = 4
        n_GPUs = 1
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            # if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

def load_image(lr_path):
    lr = cv2.imread(lr_path)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

    lr = np2Tensor(lr, 255)
    return lr

def np2Tensor(l, rgb_range):
    np_transpose = np.ascontiguousarray(l.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor
