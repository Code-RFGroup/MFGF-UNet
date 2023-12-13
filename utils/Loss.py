import torch
from torch import nn
from einops import rearrange,repeat
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def Ignore_Index(y_hat,y,ignore_index):
    y_hat_copy = rearrange(y_hat, 'b c h w -> (b h w) c')
    y_copy = rearrange(y, 'b h w -> (b h w) 1')
    mask = (y_copy != ignore_index).view(-1)
    y_hat_copy = y_hat_copy[mask]
    y_copy = y_copy[mask]
    ##
    # y_hat_copy.shape = [n,c]
    # y_copy.shape = [n,1]
    return y_hat_copy,y_copy

class BCELossWithIgnoreIndex(nn.Module):
    def __init__(self,ignore_index=-100,size_average=True):
        super(BCELossWithIgnoreIndex, self).__init__()
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self,y_hat,y):
        # y_hat:预测标签，已经过sigmoid处理 shape is (batch_size, 1,H,W)
        # y：真实标签（一般为0或1） shape is (batch_size)
        y_hat = torch.cat((1 - y_hat, y_hat), 1)  # 将二种情况的概率都列出，y_hat形状变为(batch_size, 2,H,W)
        y_hat_copy,y_copy = Ignore_Index(y_hat,y,self.ignore_index)
        result = torch.clamp(y_hat_copy.gather(1, y_copy),min=1e-7,max=1-1e-7)
        # 按照y标定的真实标签，取出预测的概率，来计算损失
        if self.size_average:
            return - torch.log(result).mean()
        else:
            return - torch.log(result)


# ------- 1. define loss function --------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.5,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.bceloss = BCELossWithIgnoreIndex(ignore_index=ignore_index,size_average=False)

    def forward(self, y_hat, y):
        BCE_loss = self.bceloss(y_hat, y)
        y_hat_copy,y_copy = Ignore_Index(y_hat,y,self.ignore_index)
        alpha_copy = rearrange(self.alpha,'c -> 1 c')
        alpha_copy = repeat(alpha_copy,'n c -> (repeat n) c',repeat=y_copy.shape[0])
        alpha_copy = alpha_copy.to(y_copy.device)
        at = alpha_copy.gather(1, y_copy)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

##########
# IOU

def _iou(pred, target,batch_size=1):
    #compute the IoU of the foreground
    Iand1 = torch.sum(target*pred)
    Ior1 = torch.sum(target) + torch.sum(pred)-Iand1
    IoU1 = Iand1/Ior1

    #IoU loss is (1-IoU1)
    IoU = 1-IoU1

    return IoU/batch_size

class IOU(torch.nn.Module):
    def __init__(self,ignore_index=-100,size_average = True):
        super(IOU, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, y_hat, y):
        y_hat_copy,y_copy=Ignore_Index(y_hat,y,self.ignore_index)
        return _iou(y_hat_copy, y_copy,batch_size=y.shape[0])

###########
# SSIM
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, nan_mask,size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map[nan_mask==False].mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, ignore_index = -100 ,size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        label = img2.float()
        label = rearrange(label,'b h w -> b 1 h w')
        nan_mask = (label==self.ignore_index)
        label[nan_mask]=0

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # print(img1.shape,label.shape)
        return _ssim(img1, label, window, self.window_size, channel,nan_mask, self.size_average)

####
#Log-Cosh Tversky Loss function (LCTLoss)
class LCTLoss(torch.nn.Module):
    def __init__(self,ignore_index=-100):
        super(LCTLoss, self).__init__()
        self.ignore_index = ignore_index
    def cross_entropy(self,x,y):
        return -1.*(y*torch.log(torch.clamp(x,min=1e-7, max=1 - 1e-7))+(1-y)*torch.log(torch.clamp(1-x,min=1e-7, max=1 - 1e-7)))

    def forward(self, y_hat, y):
        y_hat_copy,y_copy=Ignore_Index(y_hat,y,self.ignore_index)
        TverskyIndex = y_hat_copy*y_copy/(y_hat_copy*y_copy+0.7*(1-y_copy)*y_hat_copy+0.3*y_copy*(1-y_hat_copy))
        TverskyIndex = torch.clamp(TverskyIndex, min=1e-7, max=1 - 1e-7)
        loss1 = torch.log(torch.clamp(torch.cosh(TverskyIndex),min=1e-7, max=1 - 1e-7))
        loss2 = self.cross_entropy(y_hat_copy,y_copy)

        return ((loss1+loss2)/2.).mean()

if __name__ == '__main__':
    bceloss=LCTLoss(ignore_index=-1)
    pred = torch.tensor([[[[0.4,0.7],[0.3,0.4]]],[[[0.6,0.7],[0.8,0.4]]]],dtype=torch.float32)
    label = torch.tensor([[[[1,0],[0,0]]],[[[1,1],[0,-1]]]])
    loss=bceloss(pred,label)
    print(loss)
    # loss.backward()