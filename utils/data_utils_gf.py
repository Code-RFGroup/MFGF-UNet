import os
import rasterio
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

class DataProcess:
    def __init__(self,file,names,patch_size,img_size,overlap):
        self.imgs = []
        self.labels = []
        for name in tqdm(names):
            im=rasterio.open(os.path.join(file+"/data", name)).read()
            im = np.nan_to_num(im)
            la=rasterio.open(os.path.join(file+"/labels", name.replace('.jpg','_gt.png'))).read(1)
            im,la = self.process_data(im,la,patch_size,img_size,overlap)
            self.imgs += im
            self.labels += la

    def calc_start_dot(self,img_size, patch_size, overlap):
        '''
        该函数返回，每个片段的起始顶点
        :param img_size: 图像原尺寸
        :param patch_size: 分片尺寸
        :param overlap:  重叠尺寸
        :return:
        '''
        start_dot = [0]

        while True:
            dot = start_dot[-1] + patch_size - overlap
            if dot >= img_size:
                break
            elif dot < img_size:
                if dot + patch_size < img_size:
                    start_dot.append(dot)
                else:
                    start_dot.append(img_size - patch_size)
                    break
        if start_dot[0] == start_dot[-1]:
            return [start_dot[0]]        

        return start_dot

    def preprocess(self, imgs):
        '''
        对img进行Normalize预处理
        :param img:shape=[channel,W,H]
        :return:处理后的img
        '''
        means=[0.355523, 0.372792, 0.383372]
        stds=[0.139228, 0.132844, 0.126012]
        imgs = torch.tensor(imgs)
        return transforms.Normalize(means,stds)(imgs)

    def guiyihua(self,img):
        minvs=np.array([0.0, 0.0, 0.0],dtype=np.float32)
        maxvs=np.array([255.0, 255.0, 255.0],dtype=np.float32)

        return (img-minvs.reshape((-1,1,1)))/(maxvs-minvs).reshape((-1,1,1))

    def process_data(self,image, label, patch_size,img_size,overlap):
        res_image = []
        res_label = []

        image = self.guiyihua(image)
        label[label==255]=1
        label[label==-1]=255

        n = self.calc_start_dot(img_size, patch_size, overlap)
        for i in n:
            for j in n:
                im = image[:,i:i+patch_size,j:j+patch_size]
                label_ = label[i:i+patch_size,j:j+patch_size]
                res_image.append(self.preprocess(im))
                res_label.append(torch.tensor(np.array(label_)).type(torch.LongTensor))
        return res_image,res_label

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    def __len__(self):
        return len(self.imgs)