# -*- coding: utf-8 -*

from flyai.processor.base import Base
import os
from path import DATA_PATH

'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):


    def input_x(self, lr_image_path):
        image_path = os.path.join(DATA_PATH, lr_image_path)
        return image_path

    def input_y(self, hr_image_path):
        label_path = os.path.join(DATA_PATH, hr_image_path)
        return label_path

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''
    def output_y(self, pred_label):
        return pred_label
