import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Constants
pi = 3.141592653589793


# Define RGB_NewSpace class

class RGB_THSV(nn.Module):
    def __init__(self):
        super(RGB_THSV, self).__init__()
        
        # Define trainable parameters
        self.hue_param = torch.nn.Parameter(torch.full([1], 0.5))
        self.saturation_param = torch.nn.Parameter(torch.full([1], 0.5))
        self.value_param = torch.nn.Parameter(torch.full([1], 0.5))
    
    
    def RGB_to_THSV(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        batch_size, channels, height, width = img.shape

        hue = torch.zeros(batch_size, height, width, device=device, dtype=dtypes)
        saturation = torch.zeros(batch_size, height, width, device=device, dtype=dtypes)
        brightness = torch.zeros(batch_size, height, width, device=device, dtype=dtypes)

        max_val, _ = img.max(dim=1)
        min_val, _ = img.min(dim=1)

        brightness = max_val

        delta = max_val - min_val

        saturation = delta / (max_val + eps)

        mask = (delta != 0)
        hue[mask * (img[:, 0] == max_val)] = ((img[:, 1] - img[:, 2]) / (delta + eps))[mask * (img[:, 0] == max_val)]
        hue[mask * (img[:, 1] == max_val)] = (2.0 + (img[:, 2] - img[:, 0]) / (delta + eps))[
            mask * (img[:, 1] == max_val)]
        hue[mask * (img[:, 2] == max_val)] = (4.0 + (img[:, 0] - img[:, 1]) / (delta + eps))[
            mask * (img[:, 2] == max_val)]

        hue = (hue / 6.0) % 1.0
        hue[delta == 0] = 0

        hue = hue * self.hue_param
        saturation = saturation * self.saturation_param
        brightness = brightness * self.value_param
        hue = hue.unsqueeze(1)
        #print("hue shape:", hue.size())
        saturation = saturation.unsqueeze(1)
        #print("s shape:", saturation.size())
        brightness = brightness.unsqueeze(1)
        #print("b shape:", brightness.size())
        
        HSV = torch.cat([hue, saturation, brightness], dim=1)
        return HSV
    
    def THSV_to_RGB(self, new_space):
        eps = 1e-8
        device = new_space.device
        dtypes = new_space.dtype
        
        hue = new_space[:, 0, :, :]
        saturation = new_space[:, 1, :, :]
        brightness = new_space[:, 2, :, :]
        
        hue = hue / self.hue_param
        saturation = saturation / self.saturation_param
        brightness = brightness / self.value_param
        
        hue = hue * 6.0
        hi = torch.floor(hue) % 6
        f = hue - torch.floor(hue)
        
        p = brightness * (1 - saturation)
        q = brightness * (1 - f * saturation)
        t = brightness * (1 - (1 - f) * saturation)
        
        hi = hi.long()
        r = torch.zeros_like(hue)
        g = torch.zeros_like(hue)
        b = torch.zeros_like(hue)
        
        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5
        
        r[hi0] = brightness[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = brightness[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = brightness[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = brightness[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = brightness[hi4]
        
        r[hi5] = brightness[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        rgb = torch.stack([r, g, b], dim=1)
        return rgb