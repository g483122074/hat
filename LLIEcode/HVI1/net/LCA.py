import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import *


class EMA(nn.Module):
    def __init__(self, channels, factor=36):
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
    
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 10)  # 提高初始温度

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 添加通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 16, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        
        # 添加空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 应用通道注意力机制
        ca = self.channel_attention(out)
        out = out * ca
        
        # 应用空间注意力机制
        sa = self.spatial_attention(torch.cat([out.max(dim=1, keepdim=True)[0], out.mean(dim=1, keepdim=True)], dim=1))
        out = out * sa

        out = self.project_out(out)
        return out

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SCDL(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        #x1, x2 = self.dwconv(x).chunk(2, dim=1)
        #x1 = self.Tanh(self.dwconv1(x1)) + x1
        #x2 = self.Tanh(self.dwconv2(x2)) + x2
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.project_out(x)
        return x



  
# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads,drop_out_rate=0, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = SCDL(dim,drop_out_rate=drop_out_rate) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x))
        return x
    
class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x

