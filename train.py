import os
from torch import nn
import torch
from torch.utils.data import DataLoader
from utils.data_utils import SentinelDataSet,DataProcess
from tqdm import tqdm
from utils.utils_metrics import f_score
from model.MFGF_UNet import MFGF_UNet
import argparse

def write_log(path,context):
    with open(path,'a',encoding='utf8') as f:
        f.write(context)

def save_model(model,save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model,path):
    model.load_state_dict(torch.load(path))
    return model

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='', help='root dir for dataset')
    parser.add_argument('--root_path', type=str,
                        default='workspace', help='root dir for workspace')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run(default: 50)')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='batch size (default: 16)')
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.0001)')
    parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
    parser.add_argument('--break_point', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--save_model_epoch', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--val', type=str, default='True')
    args = parser.parse_args()
    # -------------------------------#
    # Set up a workspace to store weight files and logs
    # -------------------------------#
    workspace_checkpoint = os.path.join(args.root_path, "checkpoint", args.model_name)
    if not os.path.exists(workspace_checkpoint):
        os.makedirs(workspace_checkpoint)
    if not os.path.exists(os.path.join(args.root_path, "log", args.model_name)):
        os.makedirs(os.path.join(args.root_path, "log", args.model_name))
    save_best_loss_path=os.path.join(workspace_checkpoint,'best_loss.pth')
    save_best_f1_path=os.path.join(workspace_checkpoint,'best_f1.pth')
    # -------------------------------#
    # Set up a workspace to store weight files and logs
    # -------------------------------#
    device = torch.device(args.device)

    # -------------------------------#
    # create model
    # -------------------------------#
    net=MFGF_UNet(in_chs=[15]).to(device)
    # -------------------------------#
    # Set loss function and optimizer
    # -------------------------------#
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # -------------------------------#
    # DataLoader
    # -------------------------------#
    best_loss,best_f1 = 5.0,0.0
    dp = DataProcess(args.data_path, args.patch_size, args.image_size, args.overlap)
    train_imgs, train_labels, val_imgs, val_labels = dp.next_data()
    train_data = SentinelDataSet(train_imgs,train_labels)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_data = SentinelDataSet(val_imgs, val_labels)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    for i in range(args.epochs):
        log=""
        train_f_score = 0
        val_f_score = 0
        # train
        all_train_loss = 0
        train_iter=iter(train_loader)
        for_num=0
        net.train()
        for (imgs, targets) in tqdm(train_iter):
            output = net(imgs.to(device)).to(device)
            optim.zero_grad()
            result_loss = criterion(output, targets.to(device))
            result_loss.backward()
            optim.step()
            all_train_loss += result_loss.item()
            for_num += 1
            with torch.no_grad():
                #-------------------------------#
                #   calc f_score
                #-------------------------------#
                _f_score = f_score(output, targets.to(device))
            train_f_score   += _f_score.item()
        log+="Epoch-"+str(i)+"-train loss:"+format(all_train_loss/for_num,'.6')+"train_f1:"+format(train_f_score / for_num,'.5%')
        # val
        all_val_loss=0
        test_iter=iter(test_loader)
        for_num=0
        net.eval()
        for (imgs, targets) in tqdm(test_iter):
            with torch.no_grad():
                output = net(imgs.to(device)).to(device)
                result_loss = criterion(output, targets.to(device))
                all_val_loss += result_loss.item()
                for_num += 1
                #-------------------------------#
                #   calc f_score
                #-------------------------------#
                _f_score = f_score(output, targets.to(device))
                val_f_score   += _f_score.item()

        # -------------------------------#
        #   save checkpoint
        # -------------------------------#
        val_loss = all_val_loss / for_num
        val_f1 = val_f_score / for_num
        if val_loss < best_loss:
            if os.path.exists(save_best_loss_path):
                os.remove(save_best_loss_path)
            save_model(net, save_best_loss_path)
            best_loss = val_loss
        if val_f1 > best_f1:
            if os.path.exists(save_best_f1_path):
                os.remove(save_best_f1_path)
            save_model(net, save_best_f1_path)
            best_f1 = val_f1
        # -------------------------------#
        #   write log
        # -------------------------------#
        log += "val loss:" + format(val_loss,'.6')+"val_f1:"+format(val_f1,'.5%')+'\n'
        write_log(os.path.join(args.root_path, "log", args.model_name,'log.txt'), log)
