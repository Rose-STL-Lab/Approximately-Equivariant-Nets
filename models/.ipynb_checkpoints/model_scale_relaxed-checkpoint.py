import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .model_scale_equ import steerable_A, normalize_basis_by_min_scale
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###########################################################    
########### Relaxed Scale Steerable Convolution ###########
########################################################### 

class Relaxed_SESConv_Z2_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: Z2 -> (S x Z2)
    [B, C, H, W] -> [B, C', S, H', W']
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=True, basis_type='A', relu = True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.relu = relu

        basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.num_funcs, kernel_size, kernel_size).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.stdv = math.sqrt(1. / (kernel_size * kernel_size * in_channels * self.num_funcs))
        
        #self.norm = nn.BatchNorm3d(out_channels)
        self.reset_parameters()

    # def reset_parameters(self):
    #     self.weight.data.uniform_(-self.stdv, self.stdv)
    #     self.bias.data.uniform_(-self.stdv, self.stdv)
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5/25)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

            
    def get_weight_constraint(self):
        return torch.mean(torch.abs(torch.roll(self.weight, shifts=(1, 1), dims = (-2, -1)) - self.weight))
    
    def forward(self, x):
        kernel = torch.einsum("abcde, csde -> absde", self.weight, self.basis)
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # convolution
        y = F.conv2d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, self.num_scales, H, W)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        if self.relu:
            return F.relu(y)
        else:
            return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)

class Relaxed_SESConv_H_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: (S x Z2) -> (S x Z2)
    [B, C, S, H, W] -> [B, C', S', H', W']
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        scale_size: Size of scale filter
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, scale_size, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=True, basis_type='A', final_layer = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.final_layer = final_layer

        if basis_type == 'A':
            basis = steerable_A(kernel_size, scales, effective_size, )
        elif basis_type == 'B':
            basis = steerable_B(kernel_size, scales, effective_size, )

        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, scale_size, self.num_funcs, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.stdv = math.sqrt(1. / (kernel_size * kernel_size * in_channels * self.num_funcs))
        self.reset_parameters()
        
    # def reset_parameters(self):
    #     self.weight.data.uniform_(-self.stdv, self.stdv)
    #     self.bias.data.uniform_(-self.stdv, self.stdv)
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5/25)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def get_weight_constraint(self):
        return torch.mean(torch.abs(torch.roll(self.weight, shifts=(1, 1), dims = (-2, -1)) - self.weight))

    def forward(self, x):
        # get kernel
        kernel = torch.einsum("abzcde, csde -> abzsde", self.weight, self.basis)
        kernel = kernel.view(self.out_channels, self.in_channels, self.scale_size,
                             self.num_scales, self.kernel_size, self.kernel_size)

        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.scale_size,
                             self.kernel_size, self.kernel_size)

        # calculate padding
        if self.scale_size != 1:
            value = x.mean()
            x = F.pad(x, [0, 0, 0, 0, 0, self.scale_size - 1])

        output = 0.0
        for i in range(self.scale_size):
            x_ = x[:, :, i:i + self.num_scales]
            # expand X
            B, C, S, H, W = x_.shape
            x_ = x_.permute(0, 2, 1, 3, 4).contiguous()
            x_ = x_.view(B, -1, H, W)
            output += F.conv2d(x_, kernel[:, :, i], padding=self.padding,
                               groups=S, stride=self.stride)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        if self.final_layer:
            return output
        else:
            return F.relu(output)

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)



class Relaxed_Scale_SteerCNNs(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, num_layers, scales=[1.0], basis_type='A', alpha = 1e-5):
        super(Relaxed_Scale_SteerCNNs, self).__init__()
        self.model = [Relaxed_SESConv_Z2_H(in_channels, hidden_dim, kernel_size, 7, scales=scales, padding=kernel_size // 2, bias=True, basis_type='A')]
        self.model += [Relaxed_SESConv_H_H(hidden_dim, hidden_dim, 1, kernel_size, 7, scales=scales, padding=kernel_size // 2, bias=True, basis_type='A') for i in range(num_layers - 2)]
        self.model += [Relaxed_SESConv_H_H(hidden_dim, out_channels, 1, kernel_size, 7, scales=[1.0], padding=kernel_size // 2, bias=True, basis_type='A', final_layer = True)]
        self.model = nn.Sequential(*self.model)
        self.alpha = alpha
        
    def get_weight_constraint(self):
        return self.alpha * sum([layer.get_weight_constraint() for layer in self.model])

    def forward(self, x):
        return self.model(x).squeeze(2)
    

###########################################################   
############# Relaxed Scale Group Convolution #############
###########################################################      
    
class Relaxed_Scale_GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_filters, sout = 5):
        super(Relaxed_Scale_GroupConv2d, self).__init__()
        self.out_channels= out_channels
        self.in_channels = in_channels
        self.sout = sout
        self.kernel_size = kernel_size
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.combination_weights = nn.Parameter(torch.ones(self.sout, num_filters)/num_filters).to(device)
        self.weights = nn.Parameter(torch.Tensor(num_filters, out_channels, 2, in_channels//2, kernel_size, kernel_size).to(device))
        self.stdv = math.sqrt(1. / (kernel_size * kernel_size * in_channels))
        self.reset_parameters()
        self.kernels = self.kernel_generation()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def reset_parameters(self):
        self.weights.data.uniform_(-self.stdv, self.stdv)
        #self.combination_weights.data.uniform_(-self.stdv, self.stdv)
        if self.bias is not None:
            self.bias.data.fill_(0)
           
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
        return new_kernel[:,:,-kernel.shape[2]:]
    
    
    def kernel_generation(self):
        relaxed_weights = torch.einsum("nf, fabcde -> nabcde", self.combination_weights, self.weights)
        #print(torch.sum(torch.isnan(self.combination_weights)), torch.sum(torch.isnan(self.weights)))
        out = []
        for s in range(self.sout):
            if (s - self.sout//2) < 0:
                new_kernel = self.shrink_kernel(relaxed_weights[s], (self.sout//2 - s)/2).to(device)
                
            elif (s - self.sout//2) > 0:
                new_kernel = self.dilate_kernel(relaxed_weights[s], (s - self.sout//2)/2).to(device)
                
            else:
                new_kernel = relaxed_weights[s].to(device)
            
            new_kernel = new_kernel.transpose(1,2)
            new_kernel = new_kernel.reshape(new_kernel.shape[0], -1, new_kernel.shape[3], new_kernel.shape[4])
            out.append(new_kernel.detach())
        return out
    
    def forward(self, xx, level):
        padding = ((self.kernels[level].shape[-2]-1)//2, (self.kernels[level].shape[-1]-1)//2)
        conv = F.conv2d(xx, self.kernels[level].to(xx.device), padding = padding)
        out = F.relu(self.batchnorm(conv))
        return out, level
    
class gaussain_blur(nn.Module):
    def __init__(self, size, sigma, dim, channels):
        super(gaussain_blur, self).__init__()
        self.kernel = self.gaussian_kernel(size, sigma, dim, channels).to(device)
        
    def gaussian_kernel(self, size, sigma, dim, channels):

        kernel_size = 2*size + 1
        kernel_size = [kernel_size] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, channels, 1, 1, 1)

        return kernel
    
    def forward(self, xx):
        xx = xx.unsqueeze(1)
        xx = F.conv3d(xx, self.kernel, padding = (self.kernel.shape[-1]-1)//2)
        
        return xx.squeeze(1)
    
class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
    
class Relaxed_Scale_GroupConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, num_layers, num_scales, num_filters):
        super(Relaxed_Scale_GroupConvNet, self).__init__()
        self.num_scales = num_scales
        
        layers = [Relaxed_Scale_GroupConv2d(in_channels, hidden_dim, kernel_size = kernel_size, num_filters = num_filters, sout = num_scales)]
        layers += [Relaxed_Scale_GroupConv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, num_filters = num_filters, sout = num_scales) for i in range(num_layers - 2)]
        self.model = mySequential(*layers)
        self.final_layer = nn.Conv2d(hidden_dim*num_scales, out_channels, kernel_size = kernel_size, padding=(kernel_size-1)//2)
        
    
    def blur_input(self, xx): 
        out = []
        for s in np.linspace(-1, 1, 3):
            if s > 0:
                blur = gaussain_blur(size = np.ceil(s), sigma = [s**2, s, s], dim  = 3, channels = 1).to(device)
                out.append(blur(xx).unsqueeze(1)*(s+1))
            elif s<0:
                out.append(xx.unsqueeze(1)*(1/(np.abs(s)+1)))
            else:
                out.append(xx.unsqueeze(1))
        out = torch.cat(out, dim = 1)
        return out.transpose(0,1)

    def forward(self, xx):
        xx = self.blur_input(xx)
        out = torch.cat([self.model(xx[l], l)[0] for l in range(self.num_scales)], dim = 1)
        out = self.final_layer(out)
        return out
    
    
    
    
    

###########################################################################   
############# Relaxed Scale and Translation Steerable Convolution #############
###########################################################################     

class Relaxed_TS_SteerConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_size, kernel_size, effective_size, num_banks,
                 scales=[1.0], stride=1, padding=0, bias=True, basis_type='A', first_layer = False, last_layer = False):
        
        super(Relaxed_TS_SteerConv, self).__init__()
        if first_layer:
            self.convs = nn.Sequential(*[Relaxed_SESConv_Z2_H(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, 
                                                              effective_size = effective_size, scales=scales, stride=1, 
                                                              padding=padding, bias=True, basis_type='A', relu = False).to(device) for i in range(num_banks)])
        else:
            self.convs = nn.Sequential(*[Relaxed_SESConv_H_H(in_channels = in_channels, out_channels = out_channels, scale_size = scale_size, kernel_size = kernel_size, 
                                                              effective_size = effective_size, scales=scales, stride=1, 
                                                              padding=padding, bias=True, basis_type='A', final_layer = True).to(device) for i in range(num_banks)])
        
        self.combination_weights = nn.Parameter(torch.ones(64, 64, num_banks).float().to(device)/num_banks)
        
        #self.activation = nn.ReLU()
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size-1)//2
        # self.h_size = h_size
        # self.w_size = w_size
        self.last_layer = last_layer
        self.num_banks = num_banks
        
    def get_weight_constraint(self):
        return sum([layer.get_weight_constraint() for layer in self.convs])
            
    def forward(self, x):
        #print(x.shape)
        outs = torch.stack([self.convs[i](x) for i in range(self.num_banks)], dim  = 0)
        #print(outs.shape)
        # Compute Convolution
        out = torch.einsum("ijr, rbonij -> bonij", self.combination_weights, outs)

        if self.last_layer:
            return out
        else:
            return F.relu(out)
        
        

class Relaxed_TS_SteerCNNs(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, num_banks, num_layers, scales=[1.0], basis_type='A', alpha = 1e-5):
        super(Relaxed_TS_SteerCNNs, self).__init__()
        self.model = [Relaxed_TS_SteerConv(in_channels, hidden_dim, 1, kernel_size, 7, num_banks, scales=scales, padding=kernel_size // 2, 
                                           bias=True, basis_type='A', first_layer = True, last_layer = False)]
        
        self.model += [Relaxed_TS_SteerConv(hidden_dim, hidden_dim, 1, kernel_size, 7, num_banks, scales=scales, padding=kernel_size // 2, 
                                            bias=True, basis_type='A', first_layer = False, last_layer = False) for i in range(num_layers - 2)]
        
        self.model += [Relaxed_TS_SteerConv(hidden_dim, out_channels, 1, kernel_size, 7, num_banks, scales=[1.0], padding=kernel_size // 2, 
                                            bias=True, basis_type='A', first_layer = False, last_layer = True)]
        self.model = nn.Sequential(*self.model)
        self.alpha = alpha
        
    def get_weight_constraint(self):
        return self.alpha * sum([layer.get_weight_constraint() for layer in self.model])

    def forward(self, x):
        return self.model(x).squeeze(2)    
    
    
    