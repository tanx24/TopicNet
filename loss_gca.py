from torch import nn
import torch
import torch.nn.functional as F
import pytorch_toolbelt.losses as PTL

class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0,b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
            Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
            IoU1 = Iand1/Ior1

            IoU = IoU + (1-IoU1)
            
        return IoU

class DSLoss_IoU_noCAM(nn.Module):
    def __init__(self):
        super(DSLoss_IoU_noCAM, self).__init__()
        self.iou = IoU_loss()
        self.FL = PTL.BinaryFocalLoss()
    def forward(self, scaled_preds, gt):
        loss = 0
        for pred_lvl in scaled_preds[0:]:
            loss += self.iou(pred_lvl, gt)
        return loss

class ProjCos(nn.Module):
    def __init__(self):
        super(ProjCos, self).__init__()

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
           
        CosLoss = F.cosine_similarity(pred, target, dim=0)

        return CosLoss

class ModiPixelContrastLossCos(nn.Module):
    def __init__(self, temperature=0.07):
        super(ModiPixelContrastLossCos, self).__init__()
        
        self.temperature = temperature
        self.FL = ProjCos()

    def forward(self, proj_pos1, proj_pos2, proj_neg):
        similarity_pos, similarity_neg = [], []

        anchor1 = proj_pos1[-1]
        anchor2 = proj_pos2[-1]

        for iinum in range(len(proj_pos2) - 1):
            # if inum is not iinum:
            tmp1 = torch.exp(self.FL(anchor1, proj_pos1[iinum]) / self.temperature)
            tmp2 = torch.exp(self.FL(anchor2, proj_pos2[iinum]) / self.temperature)
            similarity_pos.append(tmp1)
            similarity_pos.append(tmp2)
            similarity_neg_v1 = tmp1
            similarity_neg_v2 = tmp2
            for iiinum in range(len(proj_pos1)):
                similarity_neg_v1 = similarity_neg_v1 + torch.exp(self.FL(anchor1, proj_pos2[iiinum])/ self.temperature)
                similarity_neg_v2 = similarity_neg_v2 + torch.exp(self.FL(anchor2, proj_pos1[iiinum])/ self.temperature)
            for iiinum in range(len(proj_neg)):
                similarity_neg_v1 = similarity_neg_v1 + torch.exp(self.FL(anchor1, proj_neg[iiinum])/ self.temperature)
                similarity_neg_v2 = similarity_neg_v2 + torch.exp(self.FL(anchor2, proj_neg[iiinum])/ self.temperature)
            similarity_neg.append(similarity_neg_v1)
            similarity_neg.append(similarity_neg_v2)

        loss = 0

        for inum in range(len(similarity_pos)):
            loss_partial = -torch.log(similarity_pos[inum] / similarity_neg[inum])
            loss += torch.mean(loss_partial)

        return loss

class FLLoss(nn.Module):
    def __init__(self):
        super(FLLoss, self).__init__()

        self.pcl = ModiPixelContrastLossCos()

    def forward(self, proj_pos, proj_neg):
        bp = int(proj_pos[0].shape[0] / 2)

        proj_pos1, proj_pos2 = [], []

        for inum in range(len(proj_pos)):
            proj_pos1.append(proj_pos[inum][0:bp])
            proj_pos2.append(proj_pos[inum][bp:bp*2])
        
        num_cls = int(proj_neg[0].shape[0] / bp)
        proj_negn = []
        for iinum in range(len(proj_neg)):
            for inum in range(num_cls): #7
                proj_negn.append(proj_neg[iinum][(inum*bp):(inum*bp+bp)])

        loss = self.pcl(proj_pos1, proj_pos2, proj_negn)
    
        return loss