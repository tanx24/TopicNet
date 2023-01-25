from PIL import Image
from dataset_gco import get_loader
import torch
from torchvision import transforms
from util_share import save_tensor_img, Logger
from tqdm import tqdm
from torch import nn
import os
from criterion_share import Eval
import argparse
import numpy as np
from models.TopicNet import TopicNet
import scipy.io as sio

def main(args):
    # Init model

    device = torch.device("cuda")

    model = eval('GCoNet()')
    model = model.to(device)

    topic_dict = torch.load(os.path.join('./TanNet/tmp/', 'final_topicnet.pth'))
    
    model.to(device)
    model.topic.load_state_dict(topic_dict['state_dict'])

    model.eval()
    model.set_mode('test')

    tensor2pil = transforms.ToPILImage()

    for testset in ['CoCA', 'CoSOD3k', 'CoSal2015']:
        if testset == 'CoCA':
            test_img_path = './data/images/CoCA/'
            test_gt_path = './data/gts/CoCA/'
            saved_root = os.path.join(args.save_root, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = './data/images/CoSOD3k/'
            test_gt_path = './data/gts/CoSOD3k/'
            saved_root = os.path.join(args.save_root, 'CoSOD3k')
        elif testset == 'CoSal2015':
            test_img_path = './data/images/CoSal2015/'
            test_gt_path = './data/gts/CoSal2015/'
            saved_root = os.path.join(args.save_root, 'CoSal2015')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)
        
        test_loader = get_loader(
            test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            
            scaled_preds = model(inputs)[-1]

            name = subpaths[0][0].split('/')
            grttxt.write('%d'%nnum + ': ' + name[0])
            grttxt.write('\n')
            nnum += 1

            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))
            print(scaled_preds.shape)

if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet04',
                        type=str,
                        help="Options: '', ''")
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--param_root', default='tmp_rua', type=str, help='model folder')
    parser.add_argument('--save_root', default='./results', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)



