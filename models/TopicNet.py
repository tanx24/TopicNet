import torch
from torch import nn
import torch.nn.functional as F
from models.vgg import VGG_Backbone
import numpy as np
import torch.optim as optim
from torchvision.models import vgg16
# import fvcore.nn.weight_init as weight_init
import copy
import math

class DecoderLayers(nn.Module):
    def __init__(self, in_channel=64):
        super(DecoderLayers, self).__init__()

        self.cvtlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.convlayer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.outlayer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0), 
            nn.Sigmoid()
        )
    

    def forward(self, prelayer, curlayer):
        [_, _, H, W] = curlayer.size()
        cvt_cur = self.cvtlayer(curlayer) + F.interpolate(prelayer, size=(H, W), mode='bilinear', align_corners=True)
        conv = self.convlayer(cvt_cur)
        pred = self.outlayer(conv)
        pred = self.predlayer(pred)

        return conv, pred

class ProjLayers(nn.Module):
    def __init__(self, in_channel=512):
        super(ProjLayers, self).__init__()
        self.projlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel/4), 1, kernel_size=1, stride=1, padding=0), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.projlayer(x)
        x = self.predlayer(x)
        return x

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_graph):
        nodes_q = self.query(input_graph)
        nodes_k = self.key(input_graph)
        nodes_v = self.value(input_graph)

        nodes_q_t = self.transpose_for_scores(nodes_q)
        nodes_k_t = self.transpose_for_scores(nodes_k)
        nodes_v_t = self.transpose_for_scores(nodes_v)

        attention_scores = torch.matmul(nodes_q_t, nodes_k_t.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores 

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        nodes_new = torch.matmul(attention_probs, nodes_v_t)
        nodes_new = nodes_new.permute(0, 2, 1, 3).contiguous()
        new_nodes_shape = nodes_new.size()[:-2] + (self.all_head_size,)
        nodes_new = nodes_new.view(*new_nodes_shape)
        return nodes_new

class GPPLayer(nn.Module):
    def __init__(self, config):
        super(GPPLayer, self).__init__()
        self.mha = MultiHeadAttention(config)

        self.fc_in = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_in = BertLayerNorm(config.hidden_size)
        self.dropout_in = nn.Dropout(config.hidden_dropout_prob)

        self.fc_int = nn.Linear(config.hidden_size, config.hidden_size)

        self.fc_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_out = BertLayerNorm(config.hidden_size)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_graph):
        attention_output = self.mha(input_graph)
        attention_output = self.fc_in(attention_output)
        attention_output = self.dropout_in(attention_output)
        attention_output = self.bn_in(attention_output + input_graph)
        intermediate_output = self.fc_int(attention_output)
        intermediate_output = gelu(intermediate_output)
        intermediate_output = self.fc_out(intermediate_output)
        intermediate_output = self.dropout_out(intermediate_output)
        graph_output = self.bn_out(intermediate_output + attention_output)
        return graph_output

class GPPopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2

class GPP(nn.Module):
    def __init__(self, config_gat):
        super(GPP, self).__init__()
        layer = GPPLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, input_graph):
        hidden_states = input_graph
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states)
        return hidden_states

class NLGP(nn.Module):
    def __init__(self, channel):
        super(NLGP, self).__init__()
        self.inter_channel = channel
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.scale = 1.0 / (channel ** 0.5)

    def forward(self, x):
        b, c, h, w = x.size()

        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1) #b,c,hw
        x_phi = x_phi.permute(0, 2, 1).contiguous().view(-1, self.inter_channel) #bhw,c
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(1, 0, 2).contiguous().view(self.inter_channel, -1) #c,bhw
        x_g = self.conv_g(x).view(b, self.inter_channel, -1)#b,c,hw
        x_g_1 = x_g.permute(0, 2, 1).contiguous().view(-1, self.inter_channel) #bhw,c
        x_g_2 = x_g.permute(1, 0, 2).contiguous().view(self.inter_channel, -1) #c,bhw

        mul_theta_phi = torch.matmul(x_phi, x_theta).view(b,h*w,b,h*w)
        mul_theta_phi = mul_theta_phi.permute(1,3,0,2).contiguous()
        mul_theta_phi = torch.max(mul_theta_phi, -1).values

        mul_theta_phi = F.softmax(mul_theta_phi, dim=-1)

        mul_g = torch.matmul(x_g_1, x_g_2).view(b,h*w,b,h*w).permute(1,3,0,2).contiguous()#hw,hw,b,b
        mul_g = torch.max(mul_g, -1).values #hw,hw,b
        mul_g = mul_g.permute(0,2,1).contiguous()#hw,b,hw
        mul_sum = torch.matmul(mul_g, mul_theta_phi)#hw,b,b
        mul_sum = mul_sum.permute(1,0,2).contiguous().view(-1,b) #bhw,b
        mul_sum = mul_sum.mean(-1) #bhw
        mul_sum = mul_sum.view(b, -1) * self.scale # b, hw
        mul_sum = F.softmax(mul_sum, dim=-1) # B, HW
        mul_sum = mul_sum.view(b, h, w).unsqueeze(1) # B, 1, H, W

        out = x * mul_sum
        out = self.conv_mask(out)
        
        return out

class TGroupProp(nn.Module):
    def __init__(self, input_channels):
        super(TGroupProp, self).__init__()

        self.nlgp = NLGP(input_channels)

        config_img= GPPopt(input_channels, 1)
        self.gpp = GPP(config_img)
    
    def forward(self, x):
        b, c, h, w = x.size()

        x = self.nlgp(x)
        x = self.gpp(x.view(b, -1, h*w).permute(0,2,1).contiguous())
        x = x.permute(0,2,1).contiguous().view(b, c, h, w)
        x = torch.mean(x, (0, 2, 3), keepdim = True).view(1, -1)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x

class CttLearn(nn.Module):
    def __init__(self, input_channels=512):
        super(CttLearn, self).__init__()

        self.tgp = TGroupProp(input_channels)

        self.proj_layer = ProjLayers(input_channels)

    def forward(self, x, x_ori):
        if self.training:
            div_group = 2
            bp = int(x.shape[0] / div_group)

            x_ori_list, x_pos, x_group = [], [], []
            
            for inum in range(div_group):
                tmp_x = x[(bp*inum):((bp*(inum+1)))]
                tmp_ori_x = x_ori[(bp*inum):((bp*(inum+1)))]
                tmp_group = self.tgp(tmp_x)
                tmp_pos = tmp_ori_x * tmp_group
                x_ori_list.append(tmp_ori_x)
                x_group.append(tmp_group)
                x_pos.append(tmp_pos)
            
            weighted_x = torch.cat([x_pos[i] for i in range(len(x_pos))], dim=0)

            x_neg1, x_neg2, x_neg3 = [], [], []

            for inum in range(div_group):
                for iinum in range(div_group):
                    if inum is not iinum:
                        tmp_neg = x_ori_list[inum] * x_group[iinum]
                        x_neg1.append(tmp_neg)
            
            for inum in range(div_group):
                for iinum in range(inum, div_group):
                    if inum is not iinum:
                        tmp_neg = x_pos[inum] + x_pos[iinum]
                        x_neg2.append(tmp_neg)
            
            for inum in range(div_group):
                for iinum in range(len(x_neg1)):
                    tmp_neg = x_pos[inum] + x_neg1[iinum]
                    x_neg3.append(tmp_neg)
            
            x_neg = x_neg1 + x_neg2 + x_neg3

            neg_x = torch.cat([x_neg[i] for i in range(len(x_neg))], dim=0)

            proj_pos = self.proj_layer(weighted_x)
            proj_neg = self.proj_layer(neg_x)

            proj_pos = F.interpolate(proj_pos,size=(14,14),mode='bilinear',align_corners=True)
            proj_neg = F.interpolate(proj_neg,size=(14,14),mode='bilinear',align_corners=True)

        else:
            x_group = self.tgp(x)
            weighted_x = x_ori * x_group

            proj_pos = None
            proj_neg = None
            
        return weighted_x, proj_pos, proj_neg

class Topic(nn.Module):
    def __init__(self, mode='train'):
        super(Topic, self).__init__()
        self.backbone = VGG_Backbone()
        self.mode = mode

        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))

        self.decoder4 = DecoderLayers(in_channel=512)
        self.decoder3 = DecoderLayers(in_channel=256)
        self.decoder2 = DecoderLayers(in_channel=128)
        self.decoder1 = DecoderLayers(in_channel=64)

        self.co_x5 = CttLearn(512)
        self.co_x4 = CttLearn(512)
        self.co_x3 = CttLearn(256)

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x):
        if self.mode == 'train':
            preds = self._forward(x)
        else:
            with torch.no_grad():
                preds = self._forward(x)

        return preds

    def _forward(self, x):
        [_, _, H, W] = x.size()
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)

        x3_new = F.interpolate(x3,size=x5.size()[2:],mode='bilinear',align_corners=True)
        x4_new = F.interpolate(x4,size=x5.size()[2:],mode='bilinear',align_corners=True)

        weighted_x3, proj3_pos, proj3_neg = self.co_x3(x3_new, x3)
        weighted_x4, proj4_pos, proj4_neg = self.co_x4(x4_new, x4)
        weighted_x5, proj5_pos, proj5_neg = self.co_x5(x5, x5)

        if self.training:
            proj_pos, proj_neg = [], []

            proj_pos.append(proj3_pos)
            proj_pos.append(proj4_pos)
            proj_pos.append(proj5_pos)

            proj_neg.append(proj3_neg)
            proj_neg.append(proj4_neg)
            proj_neg.append(proj5_neg)

        ########## Up-Sample ##########
        preds = []
        p5 = self.toplayer(weighted_x5)

        p4, pred4 = self.decoder4(p5, weighted_x4)
        p3, pred3 = self.decoder3(p4, weighted_x3)
        p2, pred2 = self.decoder2(p3, x2)
        p1, pred1 = self.decoder1(p2, x1)

        preds.append(F.interpolate(pred4, size=(H, W), mode='bilinear', align_corners=True))
        preds.append(F.interpolate(pred3, size=(H, W), mode='bilinear', align_corners=True))
        preds.append(F.interpolate(pred2, size=(H, W), mode='bilinear', align_corners=True))
        preds.append(F.interpolate(pred1, size=(H, W), mode='bilinear', align_corners=True))

        if self.training:
            return preds, proj_pos, proj_neg
        else:
            return preds

class TopicNet(nn.Module):
    def __init__(self, mode='train'):
        super(TopicNet, self).__init__()
        self.topic = Topic()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.topic.set_mode(self.mode)

    def forward(self, x):
        ########## Co-SOD ############
        preds = self.topic(x)

        return preds