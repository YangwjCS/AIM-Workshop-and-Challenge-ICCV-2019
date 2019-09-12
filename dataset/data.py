import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import cv2
import numpy.random as random
trainPath = "/home/zcc/Samples/broken/game-new/dataset/Train.txt"
#testPath = "/home/ywj/game/dataset/Val.txt"
#trainPath = "./Train.txt"
#testPath = "./Train10.txt"



def make_dataset(train=True):
    hazyImages = []
    clearImages = []
   # maskImages = []
    with open( "/home/zcc/Samples/broken/game-new/dataset/Train.txt", 'r') as f:
        for line in f:
            line = line.split()
            hazyImages.append(line[1])
            clearImages.append(line[2])
           # maskImages.append(line[3])

    indices = np.arange(len(clearImages))
    np.random.shuffle(indices)
    clearShuffle = []
    hazyShuffle = []
 #   maskShuffle = []

    for i in range(len(indices)):
        index = indices[i]
        clearShuffle.append(clearImages[index])
        hazyShuffle.append(hazyImages[index])
     #   maskShuffle.append(maskImages[index])

    return clearShuffle, hazyShuffle



def gammaA(image, gamma_value):
    '''
    lum = image[:,:,0]*0.299 + image[:,:,1]*0.587 + image[:,:,2]*0.114
    avgLum = np.mean(lum)
    gamma_value = 2*(0.5+avgLum)
    '''
    gammaI = (image + 1e-10) ** gamma_value
    #print(gamma_value)
    return gammaI


def random_rot(images):
    randint = random.randint(0, 4)
    if randint == 0:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_90_CLOCKWISE)
    elif randint == 1:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_180)
    elif randint == 2:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        pass
    return images


def random_crop(images, sizeTo=256):
    w = images[0].shape[1]
    h = images[0].shape[0]
    w_offset = random.randint(0, max(0, w - sizeTo - 1))
    h_offset = random.randint(0, max(0, h - sizeTo - 1))

    for i in range(len(images)):
        images[i] = images[i][h_offset:h_offset + sizeTo, w_offset:w_offset + sizeTo, :]
    return images


def random_flip(images):
    if random.random() < 0.5:
        for i in range(len(images)):
            images[i] = cv2.flip(images[i], 1)
    if random.random() < 0.5:
        for i in range(len(images)):
            images[i] = cv2.flip(images[i], 0)
    return images


def random_color_change(ximg):
    random_color = np.ones_like(ximg)
    randint = random.randint(0, 4)
    if randint == 0:
        random_color = np.random.uniform(0.6, 1.2, size=3)
    return ximg * random_color


def random_hsv_change(ximg):
    randint = random.randint(0, 4)
    if randint == 0:
        imghsv = cv2.cvtColor(ximg, cv2.COLOR_RGB2HSV).astype("float32")
        (h, s, v) = cv2.split(imghsv)
        ss = np.random.random() * 0.7
        s = s * (ss + 0.3)
        s = np.clip(s,0,255)
        vs = np.random.random() * 0.9
        v = v * (vs + 0.8)
        v = np.clip(v,0,255)
        imghsv = cv2.merge([h,s,v])
        imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2RGB)
        return imgrgb
    else:
        return ximg


def random_bright_contrast(ximg):
    randint = random.rand()
    if randint > 0.5:
        alpha = (np.random.random()*0.8 + 0.2)
        beta = np.random.random()*150.0
        xf = ximg.astype("float32")*alpha + beta
        xf = np.clip(xf,0.0,255.0)
        ximg = xf.astype("uint8")
        return ximg
    return ximg


def image_resize(images, siezeTo=(512,512)):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], siezeTo)
    return images


def normImge(image, num=1.):
    if len(image.shape) > 2:
        for i in range(3):
            img = image[:,:,i]
            max = np.max(img)
            min = np.min(img)
            image[:, :, i] = (img - min)/(max - min + 1e-8)
    else:
        max = np.max(image)
        min = np.min(image)
        image = (image - min) / (max - min + 1e-8) * num
    return image


def Balance(im, num=1.):
    img = im.copy()

    Ir = img[:, :, 0]; Ig = img[:, :, 1]; Ib = img[:, :, 2]
    Ir = normImge(Ir); Ig = normImge(Ig); Ib = normImge(Ib)
    Irm = np.mean(Ir); Igm = np.mean(Ig); Ibm = np.mean(Ib)

    Irc = Ir + (Igm - Irm) * (1 - Ir) * Ig; Irc = normImge(Irc)
    Ibc = Ib
    img = np.stack([Irc, Ig, Ibc], 2)

    #cv2.imshow('1', img)
    #cv2.waitKey()
    #img = cv2.bilateralFilter(img, 10, 10 * 2, 10 / 2)
    R = np.sum(img[:, :, 0])
    G = np.sum(img[:, :, 1])
    B = np.sum(img[:, :, 2])

    Max = np.max([R, G, B])
    ratio = np.array([Max/R, Max/G, Max/B])

    satLevel = ratio * 0.005
    [m,n,p] = img.shape

    imgRGB_orig = np.reshape(np.float64(img), (m*n, p)).transpose([1, 0])

    for ch in range(p):
        q = [satLevel[ch], 1 - satLevel[ch]]
        tiles = quantile(imgRGB_orig[ch, :], q)
        temp = imgRGB_orig[ch, :]
        temp[temp<tiles[0]] = tiles[0]
        temp[temp>tiles[1]] = tiles[1]
        imgRGB_orig[ch, :] = temp
        bottom = np.min(imgRGB_orig[ch, :])
        top = np.max(imgRGB_orig[ch, :])
        imgRGB_orig[ch, :] = (imgRGB_orig[ch, :] - bottom) * num / (top - bottom + 1e-8)

    outval = np.zeros_like(img)
    for i in range(p):
        outval[:,:, i] = np.reshape(imgRGB_orig[i, :], (m, n))
    out = outval
    return out


def quantile(v, p):
    a = []
    for i in range(len(p)):
        idx = int(len(v) * p[i]) - 1
        idx = len(v) - 1 if idx >= len(v) else idx
        idx = 0 if idx < 0 else idx
        a.append(sorted(v)[idx])
    return a


class dehazeDataloader(Dataset):
    def __init__(self, train=True, transform=None):
        clearImages, hazyImages= make_dataset(train)
        self.images = hazyImages
        self.clearImages = clearImages
        self._transform = transform
     #   self.mask = mask

    def __getitem__(self, index):
        Ix = Image.open(self.images[index]).convert('RGB')
        Ix = np.array(Ix, dtype=np.float64) / 255.

        Jx = Image.open(self.clearImages[index]).convert('RGB')
        Jx = np.array(Jx, dtype=np.float64) / 255
        
       # mask = Image.open(self.mask[index]).convert('RGB')
       # mask= np.array(mask, dtype=np.float64) / 255
        
        images = [Ix, Jx]
        '''
        if random.rand() > 0.5:
            images = random_crop(images, 256)
         #   images = image_resize(images, (320,320))
        elif random.rand() > 0.5:
            images = random_crop(images, 200)
        else:
            images = random_crop(images, 128)
         #   images = image_resize(images, (256,256))
        '''

        #print(Ix.shape)
        images = random_crop(images, 512)
        #images = image_resize(images, (512, 512))
        images = random_rot(images)
        images = random_flip(images)

        [Ix, Jx] = images

        if self._transform is not None:
            Ix, Jx= self.transform(Ix, Jx)

        return Ix, Jx

    def __len__(self):
        return len(self.images)

    def transform(self, Ix, Jx):
        #plt.imshow(img, cmap=plt.cm.gray), plt.show()
        Ix = Ix.transpose([2, 0, 1])
        Ix = torch.from_numpy(Ix).float()

        Jx = Jx.transpose([2, 0, 1])
        Jx = torch.from_numpy(Jx).float()
        
       # mask = mask.transpose([2, 0, 1])
       # mask= torch.from_numpy(mask).float()

        return Ix, Jx


class myDataloader():
    def __init__(self):
        trainset = dehazeDataloader(train=True, transform=True)
        testset = dehazeDataloader(train=False, transform=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)

        self.trainloader = trainloader
        self.testloader = testloader

    def getLoader(self):
        return self.trainloader, self.testloader


if __name__ =="__main__":

    trainLoader = dehazeDataloader(train=True, transform=True)

    for index, (Ix, Jx) in enumerate(trainLoader):
        Ix = Ix.numpy()
        Ix = Ix.transpose([1, 2, 0])

        Jx = Jx.numpy()
        Jx = Jx.transpose([1, 2, 0])
        
       # print(mask.shape)
        
        #mask = mask.numpy()
       # mask = mask.transpose([1, 2, 0])
        
        plt.subplot(221), plt.imshow(Ix, cmap=plt.cm.gray)
 
        plt.subplot(222), plt.imshow(Jx)
        plt.show()

       # plt.imshow(mask)
        plt.show()
        
