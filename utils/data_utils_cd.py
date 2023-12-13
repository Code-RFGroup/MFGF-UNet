import os
import rasterio
from torchvision import transforms
import numpy as np
import torch

class DataProcess:
    def __init__(self,file,names,patch_size,img_size,overlap):
        self.imgs = []
        self.labels = []
        for name in names:
            im=rasterio.open(os.path.join(file+"/images", name)).read()
            im = im.astype(np.float32)
            la=rasterio.open(os.path.join(file+"/labels", name)).read(1)
            im,la = self.process_data(im,la,patch_size,img_size,overlap)
            self.imgs += im
            self.labels += la

    def calc_start_dot(self,img_size, patch_size, overlap):
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
        means=[0.044765339, 0.060734, 0.06670166, 0.1064966, 0.0502187]
        stds=[0.028596097, 0.027911071, 0.0286679, 0.0423333, 0.040143659]
        imgs = torch.tensor(imgs)
        return transforms.Normalize(means,stds)(imgs)

    def guiyihua(self,img):
        minvs=np.array([0.0,0.0,0.0,0.0,0.0],dtype=np.float32)
        maxvs=np.array([19076.0,16627.0,16141.0,17614.0,19912.879],dtype=np.float32)

        return (img-minvs.reshape((-1,1,1)))/(maxvs-minvs).reshape((-1,1,1))

    def process_data(self,image, label, patch_size,img_size,overlap):
        res_image = []
        res_label = []
        image = self.guiyihua(image)
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
