import os
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio


def write_txt(i, hazyimgs, GT, maskName, train=True):
    basepath = './'
    if train:
        path = basepath + 'Test.txt'
    else:
        path = basepath + 'Val.txt'

    with open(path, 'a') as f:
        f.write(str(i) + '\t' + hazyimgs + '\t' + GT + '\t' + maskName +'\t'+'\n')


# path of images
mainPath = '/home/ywj/game/dataset/'

imgs_train = []
imgs_test = []
groundtruth_train = []
groundTruth_test = []

ii = 0
img_path = mainPath + 'hazy/'
Gt_path = mainPath + 'clear/'

real_hazy = '/home/ywj/game/dataset/ValidationBokehFree/'
real_clear = '/home/ywj/game/Training/bokeh/'

mask ='/home/ywj/game/xianzhu/BASNet/test_data/test_results/'



print(img_path)
for path, subdirs, files in os.walk(real_hazy):
    files.sort()
    for i in range(len(files)):
        nameA = files[i]
        hazyName = real_hazy + nameA
        GtName = real_clear + nameA
        maskName =mask + nameA.split('.')[0] + '.png'

        #img = Image.open(hazyName).convert('RGB')
        #Gt = Image.open(maskName).convert('RGB')
		
        write_txt(ii, hazyName, GtName, maskName)
        ii += 1
