from torch import nn
from torch.nn import ConvTranspose2d
import torch

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2'):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                     self.epsilon).pow(0.5) * self.alpha  # [B,C,1,1]
        norm = self.gamma / \
               (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size=3,padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size,bias=False,padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, input):
        return self.conv(input)

class Inseption(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inseption, self).__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_ch, in_ch+(out_ch-in_ch)//3,kernel_size=1,padding=0),
            ConvBlock(in_ch+(out_ch-in_ch)//3, in_ch+2*(out_ch-in_ch)//3, kernel_size=3, padding=1),
            ConvBlock(in_ch+2*(out_ch-in_ch)//3, out_ch, kernel_size=3, padding=1)
        )
        print(in_ch, in_ch + (out_ch - in_ch) // 3, in_ch + 2 * (out_ch - in_ch) // 3, out_ch)

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_ch, out_ch, kernel_size=1, padding=0)
        )
        self.branch4 = ConvBlock(in_ch, out_ch, kernel_size=1, padding=0)
        self.branch5 = nn.Conv2d(
            in_ch, out_ch, (1, 9), padding=(0, 4)
        )
        self.branch6 = nn.Conv2d(
            in_ch, out_ch, (9, 1), padding=(4, 0)
        )
        self.ca = GCT(out_ch * 5)
        self.all = ConvBlock(out_ch*5, out_ch, kernel_size=1, padding=0)
    def forward(self,x):
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x6 = self.branch6(x)
        x = torch.cat([x1,x3,x4,x5,x6],dim=1)
        x = self.ca(x)
        return self.all(x)

class MaxPoolWithConv(nn.Module):
    def __init__(self,in_ch):
        super(MaxPoolWithConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch,2,stride=2,bias=False)

    def forward(self,input):
        return self.conv(input)

class SegmentationBranch(nn.Module):
    def __init__(self,ch):
        super(SegmentationBranch, self).__init__()
        self.up1 = ConvTranspose2d(ch, ch//2, 2, bias=False, stride=2)
        self.conv1 = ConvBlock(ch, ch//2)
        self.up2 = ConvTranspose2d(ch//2, ch//4, 2, bias=False, stride=2)
        self.conv2 = ConvBlock(ch//2, ch//4)
        self.up3 = ConvTranspose2d(ch//4, ch//8, 2, bias=False, stride=2)
        self.conv3 = ConvBlock(ch//4, ch//8)
        self.up4 = ConvTranspose2d(ch//8, ch//16, 2, bias=False, stride=2)
        self.conv4 = ConvBlock(ch//8, ch//16)
        self.conv5 = ConvBlock(ch//16, ch//32)
        self.conv6 = nn.Conv2d(ch//32, 2, kernel_size=1)

    def forward(self,c1,c2,c3,c4,c5):
        up_1 = self.up1(c5)
        merge1 = torch.cat([up_1, c4], dim=1)
        cc1 = self.conv1(merge1)
        up_2 = self.up2(cc1)
        merge2 = torch.cat([up_2, c3], dim=1)
        cc2 = self.conv2(merge2)
        up_3 = self.up3(cc2)
        merge3 = torch.cat([up_3, c2], dim=1)
        cc3 = self.conv3(merge3)
        up_4 = self.up4(cc3)
        merge4 = torch.cat([up_4, c1], dim=1)
        cc4 = self.conv4(merge4)
        cc5 = self.conv5(cc4)
        cc6 = self.conv6(cc5)

        return cc6

class Encoder(nn.Module):
    def __init__(self,Branches,SEBlocks,channels):
        super(Encoder, self).__init__()
        self.Branches = Branches
        self.SEBlocks = SEBlocks
        self.channels = channels
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,inputs):
        self.outputs = []
        for index,ch in enumerate(self.channels):
            self.layer_outputs=[]
            for i,(branch,inp) in enumerate(zip(self.Branches,inputs)):
                out = branch[index](inp)
                self.layer_outputs.append(out)
                if index!=len(self.channels)-1:
                    out = self.maxpool(out)
                inputs[i] = out

            self.outputs.append(self.SEBlocks[index](torch.cat(self.layer_outputs,dim=1)))

        return self.outputs

class MFGF_UNet(nn.Module):
    def __init__(self,in_chs=[9],channels=[8,16,32,64,128]):
        super(MFGF_UNet,self).__init__()
        self.Branches = nn.ModuleList()
        self.SEBlocks = nn.ModuleList()
        for branch_inch in in_chs:
            model_branch = self.make_branch(branch_inch, channels)
            self.Branches.append(model_branch)
        for ch in channels:
            self.SEBlocks.append(GCT(ch*in_chs[0]))
        self.encoder = Encoder(self.Branches,self.SEBlocks,channels)
        self.Segmentation = SegmentationBranch(channels[-1]*in_chs[0])

    def make_branch(self,in_ch, channels):
        model_branch = nn.ModuleList()
        model_branch.append(Inseption(in_ch,channels[0]*in_ch))
        for index,channel in enumerate(channels[:-1]):
            model_branch.append(
                ConvBlock(channels[index]*in_ch,channels[index+1]*in_ch))

        return model_branch

    def forward(self,x):
        inputs = [x]
        encoder_outputs = self.encoder(inputs)
        c1, c2, c3, c4, c5 = encoder_outputs
        out = self.Segmentation(c1,c2,c3,c4,c5)
        return out

if __name__=="__main__":
    model = MFGF_UNet()
    input1 = torch.rand((2,9,128,128))
    gt = torch.rand((1,1,128,128))
    out =model.forward(input1)
    print(out.shape)
