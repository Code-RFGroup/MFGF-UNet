import os
import rasterio
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch

class WIPIDataSet(Dataset):
    def __init__(self,file,patch_size,img_size,overlap,istrain=True):
        self.imgs = []
        self.labels = []
        if istrain:
            with open('./train.txt', 'r', encoding='utf8') as f:
                names = f.read().strip().split('\n')
        else:
            with open('./val.txt', 'r', encoding='utf8') as f:
                names = f.read().strip().split('\n')
        for name in names:
            im1 = rasterio.open(os.path.join(file + "/S1", name + "_S1Hand.tif")).read()
            im1 = np.nan_to_num(im1)
            im2 = rasterio.open(os.path.join(file + "/S2", name + "_S2Hand.tif")).read()
            im = np.concatenate([im1, im2], axis=0)
            la = rasterio.open(os.path.join(file + "/Labels", name + "_LabelHand.tif")).read(1)
            im, la = self.process_data(im, la, patch_size, img_size, overlap)
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
        means=[0.6135455, 0.69895583,0.600975, 0.5279132, 0.41308516, 0.35912883, 0.23359457, 0.5356049, 0.5820195]
        stds=[0.0954984, 0.109712355,0.163451, 0.10016587, 0.12622261, 0.14469956, 0.03052271, 0.06205064, 0.051749207]
        imgs = torch.tensor(imgs)
        return transforms.Normalize(means,stds)(imgs)

    def guiyihua(self,img):
        minvs=np.array([-89.78576, -103.87603,-0.5973094, -0.6403712, -1.0, -0.80392665, -16.582752, -22133.5, -1332158.2],dtype=np.float32)
        maxvs=np.array([36.832024, 17.376385,1.0, 1.0, 0.7095436, 1.0, 53.36709, 14712.5, 753355.75],dtype=np.float32)

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
        return self.imgs[idx],self.labels[idx]

    def __len__(self):
        return len(self.imgs)
