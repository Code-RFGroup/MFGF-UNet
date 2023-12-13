import torch
import rasterio
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from utils.EvalUtils import Evaluation
from model.MFGF_UNet import MFGF_UNet
import time

def calc_start_dot(img_size,patch_size,overlap):
    start_dot = [0]

    while True:
        dot=start_dot[-1] + patch_size - overlap
        if dot>=img_size:
            break
        elif dot < img_size:
            if dot+patch_size<img_size:
                start_dot.append(dot)
            else:
                start_dot.append(img_size-patch_size)
                break

    if start_dot[0]==start_dot[-1]:
        return [start_dot[0]]
    return start_dot

def load_model(model,path):
    model.load_state_dict(torch.load(path,map_location='cpu'))
    return model

def preprocess(imgs):
    means = [0.6135455, 0.69895583, 0.600975, 0.5279132, 0.41308516, 0.35912883, 0.23359457, 0.5356049, 0.5820195]
    stds = [0.0954984, 0.109712355, 0.163451, 0.10016587, 0.12622261, 0.14469956, 0.03052271, 0.06205064, 0.051749207]
    imgs = torch.tensor(imgs)
    return transforms.Normalize(means, stds)(imgs)


def guiyihua(img):
    minvs = np.array(
        [-89.78576, -103.87603, -0.5973094, -0.6403712, -1.0, -0.80392665, -16.582752, -22133.5, -1332158.2],
        dtype=np.float32)
    maxvs = np.array([36.832024, 17.376385, 1.0, 1.0, 0.7095436, 1.0, 53.36709, 14712.5, 753355.75], dtype=np.float32)

    return (img - minvs.reshape((-1, 1, 1))) / (maxvs - minvs).reshape((-1, 1, 1))

def readtiff(s1_file,s2_file,start_dot,patch_size):
    im1 = rasterio.open(s1_file).read()
    im1 = np.nan_to_num(im1)
    im2 = rasterio.open(s2_file).read()
    im = np.concatenate([im1, im2], axis=0)
    res_image = []

    image = guiyihua(im)

    for b1 in start_dot:
        for b2 in start_dot:
            im = image[:, b1:b1 + patch_size, b2:b2 + patch_size]
            res_image.append(preprocess(im))

    return res_image

def merge(imgs,start_dot,patch_size,img_size):
    pic=np.zeros((img_size,img_size))
    num_label=np.zeros((img_size,img_size))
    l=len(start_dot)
    for i,b1 in enumerate(start_dot):
        for j,b2 in enumerate(start_dot):
            img=imgs[i*l+j]
            pic[b1: b1 + patch_size,b2:b2 + patch_size]+=img
            num_label[b1: b1 + patch_size,b2:b2 + patch_size]+=1
    return pic/num_label

def read_gtiff(data_dir):
    return rasterio.open(data_dir)

def plt_pic(pic,label):
    fig, ((ax0,ax1)) = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharey=True)
    ax0.imshow(pic,cmap=plt.cm.gray)
    ax1.imshow(label,cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    overlap = 32
    patch_size = 128
    img_size = 512
    type = 'test_wipi'
    dataset_path = r'data/WIPI'
    with open(type+'.txt','r') as f:
        names = f.read().strip().split('\n')
    start_dot=calc_start_dot(img_size,patch_size,overlap)
    print(start_dot)
    device=torch.device('cuda:1')
    unet = MFGF_UNet(in_chs=[9])
    weight_path = ''
    unet =load_model(unet,weight_path)
    unet.to(device)
    unet.eval()
    ev_imgs, ev_masks = [], []
    outs=[]
    alltime=0
    for name in names:
        imgs=readtiff(dataset_path+"SAR/"+name+"_S1Hand.tif",dataset_path+"WI/"+name+"_S2Hand.tif",start_dot,patch_size)
        all_output=np.empty((0,2,patch_size,patch_size))
        start = time.time()
        for i in range(len(start_dot)):
            input = torch.stack(imgs[i * len(start_dot):(i + 1) * len(start_dot)]).to(device)
            output = unet(input.to(device))
            output = nn.Softmax(dim=1)(output).cpu().detach().numpy()
            all_output = np.concatenate((all_output, output))

        end = time.time()
        alltime += (end - start)
        label_file = dataset_path + 'Labels/' + name+'_LabelHand.tif'
        la = rasterio.open(label_file).read(1)
        pred = merge(all_output[:, 1, :, :], start_dot, patch_size, img_size)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        plt_pic(pred,la)
        pred = pred.reshape((-1,))
        la = la.reshape((-1))
        ev_imgs.append(pred)
        ev_masks.append(la)
    print('Running time: %s Seconds' % alltime)
    print(Evaluation(num_classes=2,calc_class=1).all_(ev_imgs, ev_masks))

