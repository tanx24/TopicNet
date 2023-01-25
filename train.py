import torch
import torch.nn as nn
import torch.optim as optim
from util_share import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset_gco import get_loader
from criterion_share import Eval
import torchvision.utils as vutils
from models.TopicNet import TopicNet
import torch.nn.functional as F
import pytorch_toolbelt.losses as PTL

# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='CoSalNet',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--loss',
                    default='DSLoss_IoU',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--bs', '--batch_size', default=16, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='Jigsaw2_DUTS',
                    type=str,
                    help="Options: 'Jigsaw2_DUTS', 'DUTS_class'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default=None, help='Temporary folder')

args = parser.parse_args()

train_img_path = './data/images/DUTS_class/'
train_gt_path = './data/gts/DUTS_class/'

# make dir for tmp
log_path = 'tmp/' + args.tmp
save_path = '/home/tanx/disk/2021/TanNet/'  + args.tmp

os.makedirs(log_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)


# Init log file
logger = Logger(os.path.join(log_path, "log.txt"))

set_seed(1996)

# Init model
device = torch.device("cuda")

model = eval('TopicNet()')
model = model.to(device)

backbone_params = list(map(id, model.topic.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.topic.parameters())

all_params = [{'params': base_params}, {'params': model.topic.backbone.parameters(), 'lr': args.lr * 0.01}]

# Setting optimizer
optimizer = optim.Adam(params=all_params, lr=args.lr, betas=[0.9, 0.99])

for key, value in model.named_parameters():
    if 'topic.backbone' in key and 'topic.backbone.conv5.conv5_3' not in key:
        value.requires_grad = False

# Setting Loss
exec('from loss_gca import ' + args.loss)
dsloss = eval(args.loss+'()')
exec('from loss_gca import ' + 'FLLoss')
cttloss = eval('FLLoss()')

def main():
    for epoch in range(args.start_epoch, args.epochs):
        # epoch = pro_epoch
        train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              1, #args.bs,
                              max_num=args.bs, #16, #20,
                              istrain=True,
                              shuffle=False,
                              num_workers=4, #4,
                              epoch=epoch,
                              pin=True)


        train_loss = train(epoch, train_loader)
        [val_mae1, val_Sm1, val_mae2, val_Sm2] = validate(epoch)
        
        save_checkpoint(
            {
                'state_dict': model.topic.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            path=os.path.join(save_path, "checkpoint_%d.pth"%epoch))

def train(epoch, train_loader):
    loss_log = AverageMeter()

    # Switch to train mode
    model.train()
    model.set_mode('train')

    FL = PTL.BinaryFocalLoss()

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)

        scaled_preds, proj_pos, proj_neg = model(inputs)

        loss_sal = dsloss(scaled_preds, gts)
        loss_ctt = cttloss(proj_pos, proj_neg) #*一个值。

        loss = loss_sal + loss_ctt

        loss_log.update(loss, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]  '
                        'Train Loss: loss_sal: {4:.3f}, loss_ctt: {5:.3f}, '
                        'Loss_total: {loss.val:.3f} ({loss.avg:.3f})  '.format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss_sal,
                            loss_ctt,
                            loss=loss_log,
                        ))
    logger.info('@==Final== Epoch[{0}/{1}]  '
                'Train Loss: {loss.avg:.3f}  '.format(epoch,
                                                      args.epochs,
                                                      loss=loss_log))

    return loss_log.avg

def validate(epoch):

    # Switch to evaluate mode
    val_img_path1 = './data/images/CoSal2015/'
    val_gt_path1 = './data/gts/CoSal2015/'
    val_loader1 = get_loader(
            val_img_path1, val_gt_path1, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

    val_img_path2 = './data/images/CoCA/'
    val_gt_path2 = './data/gts/CoCA/'
    val_loader2 = get_loader(
            val_img_path2, val_gt_path2, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

    model.eval()
    model.set_mode('test')

    saved_root1 = os.path.join(save_path, 'Salmaps1')
    # make dir for saving results
    os.makedirs(saved_root1, exist_ok=True)

    for batch in tqdm(val_loader1):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        subpaths = batch[2]
        ori_sizes = batch[3]
            
        scaled_preds = model(inputs)[-1]

        num = len(scaled_preds)

        os.makedirs(os.path.join(saved_root1, subpaths[0][0].split('/')[0]),
                    exist_ok=True)

        for inum in range(num):
            subpath = subpaths[inum][0]
            ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
            res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
            save_tensor_img(res, os.path.join(saved_root1, subpath))

    evaler = Eval(pred_root=saved_root1, label_root=val_gt_path1)
    mae1 = evaler.eval_mae()
    Sm1 = evaler.eval_Smeasure()

    saved_root2 = os.path.join(save_path, 'Salmaps2')
    # make dir for saving results
    os.makedirs(saved_root2, exist_ok=True)

    for batch in tqdm(val_loader2):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        subpaths = batch[2]
        ori_sizes = batch[3]
            
        scaled_preds = model(inputs)[-1]

        num = len(scaled_preds)

        os.makedirs(os.path.join(saved_root2, subpaths[0][0].split('/')[0]),
                    exist_ok=True)

        for inum in range(num):
            subpath = subpaths[inum][0]
            ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
            res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
            save_tensor_img(res, os.path.join(saved_root2, subpath))

    evaler = Eval(pred_root=saved_root2, label_root=val_gt_path2)
    mae2 = evaler.eval_mae()
    Sm2 = evaler.eval_Smeasure()

    logger.info('@==Final== Epoch[{0}/{1}]  '
                'Cosal2015: MAE: {mae1:.3f}  '
                'Sm: {Sm1:.3f}  '
                'CoCA: MAE: {mae2:.3f}  '
                'Sm: {Sm2:.3f}'.format(epoch, args.epochs, mae1=mae1, Sm1=Sm1, mae2=mae2,Sm2=Sm2))

    return mae1, Sm1, mae2, Sm2

if __name__ == '__main__':
    main()
