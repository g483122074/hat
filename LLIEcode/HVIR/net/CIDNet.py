import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from torchvision.models.convnext import LayerNorm2d

class EMA(nn.Module):
    def __init__(self, channels, factor=3):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0, "Channels must be divisible by the number of groups."
        
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(num_groups=channels // self.groups, num_channels=channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3x3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.size()
        
        group_x = x.view(b * self.groups, c // self.groups, h, w)
        
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        
        x11 = self.softmax(self.agp(x1).view(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.view(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).view(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.view(b * self.groups, c // self.groups, -1)
        
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).view(b * self.groups, 1, h, w)
        
        out = (group_x * weights.sigmoid()).view(b, c, h, w)
        
        return out


class ELA(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(ELA, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(18, channel)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
 
        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)
 
        
        # 在两个维度上应用注意力
        return x * x_h * x_w

class CIDNet(nn.Module):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI().cuda()
        self.ema=EMA(3)
        self.ela=ELA(ch1)
        
    def forward(self, x):
        dtypes = x.dtype
        x=self.ema(x)
        hvi = self.trans.HVIT(x)

        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        i_enc1=self.ela(i_enc1)

        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        hv_1=self.ela(hv_1)

        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
    

