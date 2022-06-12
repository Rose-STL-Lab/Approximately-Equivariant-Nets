import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from .model_scale_equ import SESConv_Z2_H, SESConv_H_H
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### Regular MLP ##########
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, h_size, w_size):
        super(MLPBlock, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.linear(x.view(x.shape[0], -1))
    
class MLPNet(nn.Module):
    def __init__(self, in_channels, out_channels, h_size, w_size, hidden_dim, num_layers):
        super(MLPNet, self).__init__()
        self.layers = [MLPBlock(in_channels*h_size*w_size, hidden_dim, h_size, w_size)]
        self.layers += [MLPBlock(hidden_dim, hidden_dim, h_size, w_size) for i in range(num_layers-2)]
        self.layers += [nn.Linear(hidden_dim, out_channels*h_size*w_size)]
        self.model = nn.Sequential(*self.layers)
        self.w_size = w_size
        self.h_size = h_size
        
    def forward(self, x):
        return self.model(x.view(x.shape[0], -1)).reshape(x.shape[0], -1, self.h_size, self.w_size)
         
        
########### Regular CNN ########        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ) 
        
    def forward(self, x):
        return self.conv(x)
    
class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, num_layers):
        super(ConvNet, self).__init__()
        self.layers = [ConvBlock(in_channels, hidden_dim, kernel_size)]
        self.layers += [ConvBlock(hidden_dim, hidden_dim, kernel_size) for i in range(num_layers-2)]
        self.layers += [nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=(kernel_size-1)//2)]
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.model(x)
    


############ ConvNet + SEConv ############
class ConvEquScale(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_hidden_dim, equ_hidden_dim, conv_num_layers, equ_num_layers, scales = [1.0]):
        super(ConvEquScale, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, conv_hidden_dim, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Conv2d(conv_hidden_dim, equ_hidden_dim, kernel_size, padding=(kernel_size-1)//2)
        ) 
        
        self.equnet = [SESConv_Z2_H(equ_hidden_dim, equ_hidden_dim, kernel_size, 7, scales=scales, bias=True, padding=kernel_size // 2)]
        self.equnet += [SESConv_H_H(equ_hidden_dim, equ_hidden_dim, 1, kernel_size, 7, scales=scales, bias=True,  padding=kernel_size // 2) for i in range(equ_num_layers - 2)]
        self.equnet += [SESConv_H_H(equ_hidden_dim, out_channels, 1, kernel_size, 7, scales=[1.0], bias=True, padding=kernel_size // 2, final_layer = True)]
        self.equnet = nn.Sequential(*self.equnet)
        
    def forward(self, xx):
        out = self.convnet(xx)
        out = self.equnet(out)
        return out.squeeze(2)
    
    
############ Rotational Residual Pathway ############
class Scale_RPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_scales, scales, first_layer = False, final_layer = False):
        super(Scale_RPPBlock, self).__init__()
        self.first_layer = first_layer
        self.final_layer = final_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.first_layer:
            self.conv = nn.Conv2d(in_channels, out_channels*num_scales, kernel_size, padding=(kernel_size-1)//2)
        elif self.final_layer:
            self.conv = nn.Conv2d(in_channels*num_scales, out_channels, kernel_size, padding=(kernel_size-1)//2)
        else:
            self.conv = nn.Conv2d(in_channels*num_scales, out_channels*num_scales, kernel_size, padding=(kernel_size-1)//2)
            
        if self.first_layer:
            self.equ = SESConv_Z2_H(in_channels, out_channels, kernel_size, 7, scales=scales, padding=kernel_size // 2, final_layer = False)
        elif self.final_layer:
            self.equ = SESConv_H_H(in_channels, out_channels, 1, kernel_size, 7, scales=scales, padding=kernel_size // 2, final_layer = True)
        else:
            self.equ = SESConv_H_H(in_channels, out_channels, 1, kernel_size, 7, scales=scales, padding=kernel_size // 2, final_layer = False)
        

    def forward(self, x):
        if not self.first_layer:
            convout = self.conv(x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1]))
        else:
            convout = self.conv(x)
            
        equnout = self.equ(x)
        #print(equnout.shape, convout.shape)
        convout = convout.reshape(equnout.shape)
        #.unsqueeze(-3).repeat(1, 1, equnout.shape[-3], 1, 1)
        
        if self.final_layer:
            return convout + equnout
        else:
            return F.relu(convout + equnout)
    
class Scale_RPPNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, scales, num_layers):
        super(Scale_RPPNet, self).__init__()
        self.num_scales = len(scales)
        
        self.layers = [Scale_RPPBlock(in_channels, hidden_dim, kernel_size, self.num_scales, scales, first_layer = True, final_layer = False)]
        
        self.layers += [Scale_RPPBlock(hidden_dim, hidden_dim, kernel_size, self.num_scales, scales, first_layer = False, final_layer = False) for i in range(num_layers-2)]
        
        self.layers += [Scale_RPPBlock(hidden_dim, out_channels, kernel_size, self.num_scales, [1.0], first_layer = False, final_layer = True)]
        
        self.model = nn.Sequential(*self.layers)
        
        
    def get_weight_constraint(self, conv_wd = 1e-6):
        conv_l2 = 0.
        basic_l2 = 0.
        for block in self.model:
            if hasattr(block, 'conv'):
                conv_l2 += sum([p.pow(2).sum() for p in block.conv.parameters()])
        return conv_wd*conv_l2
        
    def forward(self, x):
        return self.model(x).squeeze(2)
    
    
############ Lift_Expansion ############
class Lift_Scale_Expansion(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_hidden_dim, backbone_hidden_dim, kernel_size, num_layers, scales=[1.0]):
        super(Lift_Scale_Expansion, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*64*64, encoder_hidden_dim),
            #nn.BatchNorm1d(encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
           # nn.BatchNorm1d(encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, backbone_hidden_dim//2)
        )
            
        self.part_one = [SESConv_Z2_H(in_channels, backbone_hidden_dim, kernel_size, 7, scales=scales, padding=kernel_size // 2, bias=True, basis_type='A'),
                         SESConv_H_H(backbone_hidden_dim, backbone_hidden_dim, 1, kernel_size, 7, scales=scales, padding=kernel_size // 2, bias=True, basis_type='A')]
        self.part_one = nn.Sequential(*self.part_one)
        
        self.part_two = [SESConv_H_H(backbone_hidden_dim+backbone_hidden_dim//2, backbone_hidden_dim, 1, 
                                      kernel_size, 7, scales=scales, padding=kernel_size // 2, bias=True, basis_type='A')]
        self.part_two += [SESConv_H_H(backbone_hidden_dim, backbone_hidden_dim, 1, 
                                      kernel_size, 7, scales=scales, padding=kernel_size // 2, bias=True, basis_type='A')  for i in range(num_layers - 4)]
        self.part_two += [SESConv_H_H(backbone_hidden_dim, out_channels, 1, kernel_size, 7, scales=[1.0], padding=kernel_size // 2, bias=True, basis_type='A', final_layer = True)]
        self.part_two = nn.Sequential(*self.part_two)

    def forward(self, x):
        encoder_out = self.encoder(x.reshape(x.shape[0], -1))
        out = self.part_one(x)
        encoder_out = encoder_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out.shape[2], x.shape[-2], x.shape[-1])
        out = torch.cat([encoder_out, out], dim = 1)
        out = self.part_two(out)
        return out.squeeze(2)
    
    
############ Constrained Locally Connected NN ############
class Constrained_LCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_size, w_size, num_scales = 5, final_layer = False):
        super(Constrained_LCBlock, self).__init__()
        torch.manual_seed(0)
        self.weights = nn.Parameter(torch.randn(h_size, w_size, out_channels, in_channels, kernel_size, kernel_size).float().to(device)/in_channels)
        self.bias = nn.Parameter(torch.randn(out_channels).float().to(device)/in_channels)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size-1)//2
        self.h_size = h_size
        self.w_size = w_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_layer = final_layer
        self.sout = num_scales
        
    def rot_vector(self, inp, theta):
        #inp shape: c x 2 x 64 x 64
        theta = torch.tensor(theta).float().to(device)
        rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float().to(device)
        out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
        return out
    
    def shrink_kernel(self, kernel, up_scale):
        up_scale = torch.tensor(up_scale).float()
        pad_in = (torch.ceil(up_scale**2).int())*((kernel.shape[2]-1)//2)
        pad_h = (torch.ceil(up_scale).int())*((kernel.shape[3]-1)//2)
        pad_w = (torch.ceil(up_scale).int())*((kernel.shape[4]-1)//2)
        padded_kernel = F.pad(kernel, (pad_w, pad_w, pad_h, pad_h, pad_in, pad_in))
        delta = up_scale%1
        if delta == 0:
            shrink_factor = 1
        else:
            # shrink_factor for coordinates if the kernel is over shrunk.
            shrink_factor = (((kernel.shape[4]-1))/(padded_kernel.shape[-1]-1)*(up_scale+1))
            # Adjustment to deal with weird filtering on the grid sample function.
            shrink_factor = 1.5*(shrink_factor-0.5)**3 + 0.57   

        grid = torch.meshgrid(torch.linspace(-1, 1, kernel.shape[2])*(shrink_factor**2),
                              torch.linspace(-1, 1, kernel.shape[3])*shrink_factor, 
                              torch.linspace(-1, 1, kernel.shape[4])*shrink_factor)

        grid = torch.cat([grid[2].unsqueeze(0).unsqueeze(-1), 
                          grid[1].unsqueeze(0).unsqueeze(-1), 
                          grid[0].unsqueeze(0).unsqueeze(-1)], dim = -1).repeat(kernel.shape[0],1,1,1,1)

        new_kernel = F.grid_sample(padded_kernel, grid.to(device))
        if kernel.shape[-1] - 2*up_scale > 0:
            new_kernel = new_kernel * (kernel.shape[-1]**2/((kernel.shape[-1] - 2*up_scale)**2 + 0.01))
        return new_kernel
    
    #@staticmethod
    def dilate_kernel(self, kernel, dilation):
        if dilation == 0:
            return kernel 

        dilation = torch.tensor(dilation).float()
        delta = dilation%1

        d_in = torch.ceil(dilation**2).int()
        new_in = kernel.shape[2] + (kernel.shape[2]-1)*d_in

        d_h = torch.ceil(dilation).int()
        new_h = kernel.shape[3] + (kernel.shape[3]-1)*d_h

        d_w = torch.ceil(dilation).int()
        new_w = kernel.shape[4] + (kernel.shape[4]-1)*d_h

        new_kernel = torch.zeros(kernel.shape[0], kernel.shape[1], new_in, new_h, new_w)
        new_kernel[:,:,::(d_in+1),::(d_h+1), ::(d_w+1)] = kernel
        shrink_factor = 1
        # shrink coordinates if the kernel is over dilated.
        if delta != 0:
            new_kernel = F.pad(new_kernel, ((kernel.shape[4]-1)//2, (kernel.shape[4]-1)//2)*3)

            shrink_factor = (new_kernel.shape[-1] - 1 - (kernel.shape[4]-1)*(delta))/(new_kernel.shape[-1] - 1) 
            grid = torch.meshgrid(torch.linspace(-1, 1, new_in)*(shrink_factor**2), 
                                  torch.linspace(-1, 1, new_h)*shrink_factor, 
                                  torch.linspace(-1, 1, new_w)*shrink_factor)

            grid = torch.cat([grid[2].unsqueeze(0).unsqueeze(-1), 
                              grid[1].unsqueeze(0).unsqueeze(-1), 
                              grid[0].unsqueeze(0).unsqueeze(-1)], dim = -1).repeat(kernel.shape[0],1,1,1,1)

            new_kernel = F.grid_sample(new_kernel, grid)         
            #new_kernel = new_kernel/new_kernel.sum()*kernel.sum()
        return new_kernel[:,:,-kernel.shape[2]:]
    
    
    def get_weight_constraint(self):
        out = []
        con_weights = self.weights.reshape(-1, self.in_channels, self.kernel_size, self.kernel_size)
        con_weights = con_weights.reshape(con_weights.shape[0], self.in_channels//2, 2, self.kernel_size, self.kernel_size).transpose(1,2)
        for s in range(self.sout):
            if (s - self.sout//2) < 0:
                new_kernel = self.shrink_kernel(con_weights, (self.sout//2 - s)/2).to(device)
            elif (s - self.sout//2) > 0:
                new_kernel = self.dilate_kernel(con_weights, (s - self.sout//2)/2).to(device)
            else:
                new_kernel = con_weights.to(device)
            
            new_kernel = new_kernel.transpose(1,2)
            new_kernel = new_kernel.reshape(new_kernel.shape[0], -1, new_kernel.shape[3], new_kernel.shape[4])
            out.append(new_kernel.unsqueeze(0)[..., new_kernel.shape[-2]//2-self.kernel_size//2:new_kernel.shape[-2]//2+self.kernel_size//2, 
                                               new_kernel.shape[-1]//2-self.kernel_size//2:new_kernel.shape[-1]//2+self.kernel_size//2])
           # print(new_kernel.shape)
        out = torch.cat(out, dim = 0)    
        return torch.mean(torch.abs(torch.roll(out, shifts=1, dims = 0) - out))
        
        
    def forward(self, x):
        x = F.unfold(x, kernel_size = self.kernel_size, padding = (self.kernel_size-1)//2)
        x = x.reshape(x.shape[0], self.in_channels, self.kernel_size, self.kernel_size, -1)
        x = x.reshape(x.shape[0], self.in_channels, self.kernel_size, self.kernel_size, self.h_size, self.w_size)
        
        # Compute Convolution: h x w x o x c x k x k and bz x c x k x k x h x w
        out = torch.einsum("ijabcd, rbcdij -> raij", self.weights, x)

        
        if self.final_layer:
            return out
        else:
            return self.activation(out)
        
class Constrained_Scale_LCNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, h_size, w_size, num_layers, alpha = 1):
        super(Constrained_Scale_LCNet, self).__init__()
        layers = [Constrained_LCBlock(in_channels, hidden_dim, kernel_size, h_size, w_size)]
        layers += [Constrained_LCBlock(hidden_dim, hidden_dim, kernel_size, h_size, w_size) for i in range(num_layers-2)]
        layers += [Constrained_LCBlock(hidden_dim, out_channels, kernel_size, h_size, w_size, final_layer = True)]
        self.rconv = nn.Sequential(*layers)
        self.num_layers = num_layers
        self.alpha = alpha
        
    def get_weight_constraint(self): 
        constraint = 0
        for layer in self.rconv:
            constraint += layer.get_weight_constraint()
        return self.alpha*constraint
    
    def forward(self, x):
        return self.rconv(x)