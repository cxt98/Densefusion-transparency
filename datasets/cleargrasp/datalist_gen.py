import json
import numpy as np
from numpy.linalg import inv
import os
import scipy.io as sio
from PIL import Image
from scipy.spatial.transform import Rotation as R

# obj_to_label={'cup-with-waves': 1,
# 	      'flower-bath-bomb' : 2,
#               'heart-bath-bomb' : 3,
#               'square-plastic-bottle' : 4,
#               'stemless-plastic-champagne-glass' : 5}
	
root_train = '/home/cxt/Documents/research/lf_perception/598-007-project/cleargrasp-dataset-train/'
root_test = '/home/cxt/Documents/research/lf_perception/598-007-project/cleargrasp-testing-validation/synthetic-val/'
dst_folder = '/home/cxt/Documents/research/lf_perception/598-007-project/DenseFusion/datasets/cleargrasp/dataset_config/'

with open(dst_folder + 'train_data_list.txt', 'w') as f:
    for fold in sorted(os.listdir(root_train)):
        rgb_dir = root_train + fold + '/rgb-imgs/'
        rgb_files = sorted(os.listdir(rgb_dir))
        for fi in rgb_files:
            print(fi)
            f.writelines(fold + '/' + fi[:-8] + '\n')
            # im = Image.open(rgb_dir + '/' + fi)
            # if im.size==(1920, 1080):
            #     f.writelines(fold + '/' + fi[:-8] + '\n')

with open(dst_folder + 'test_data_list.txt', 'w') as f:
    for fold in sorted(os.listdir(root_test)):
        rgb_dir = root_test + fold + '/rgb-imgs/'
        rgb_files = sorted(os.listdir(rgb_dir))
        for fi in rgb_files:
            print(fi)
            f.writelines(fold + '/' + fi[:-8] + '\n')

