import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

###################################################    
########### Scale Steerable Convolution ###########
################################################### 

def hermite_poly(X, n):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array
    Output:
        Y: array of shape X.shape
    """
    coeff = [0] * n + [1]
    func = np.polynomial.hermite_e.hermeval(X, coeff)
    return func


def onescale_grid_hermite_gaussian(size, scale, max_order=None):
    max_order = max_order or size - 1
    X = np.linspace(-(size // 2), size // 2, size)
    Y = np.linspace(-(size // 2), size // 2, size)
    order_y, order_x = np.indices([max_order + 1, max_order + 1])

    G = np.exp(-X**2 / (2 * scale**2)) / scale

    basis_x = [G * hermite_poly(X / scale, n) for n in order_x.ravel()]
    basis_y = [G * hermite_poly(Y / scale, n) for n in order_y.ravel()]
    basis_x = torch.Tensor(np.stack(basis_x))
    basis_y = torch.Tensor(np.stack(basis_y))
    basis = torch.bmm(basis_x[:, :, None], basis_y[:, None, :])
    return basis


def steerable_A(size, scales, effective_size, **kwargs):
    max_order = effective_size - 1
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        basis = onescale_grid_hermite_gaussian(size_before_pad, scale, max_order)
        basis = basis[None, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    return torch.stack(basis_tensors, 1)


def normalize_basis_by_min_scale(basis):
    norm = basis.pow(2).sum([2, 3], keepdim=True).sqrt()[:, [0]]
    return basis / norm



class SESConv_Z2_H(nn.Module):
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
                 scales=[1.0], stride=1, padding=0, bias=True, basis_type='A', final_layer = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.final_layer = final_layer
        self.norm = nn.BatchNorm3d(out_channels)


        basis = steerable_A(kernel_size, scales, effective_size)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)
        
        

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.ones(out_channels, in_channels, self.num_funcs).float().to(device)/self.num_funcs)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
#         stdv = np.sqrt(1/(in_channels))
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels,
                             self.num_scales, self.kernel_size, self.kernel_size)
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # convolution
        y = F.conv2d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, self.num_scales, H, W)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        if self.final_layer:
            return y
        else:
            return F.relu(y)#)self.norm(

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)

class SESConv_H_H(nn.Module):
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
        self.norm = nn.BatchNorm3d(out_channels)

        if basis_type == 'A':
            basis = steerable_A(kernel_size, scales, effective_size)
        elif basis_type == 'B':
            basis = steerable_B(kernel_size, scales, effective_size)

        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.ones(
            out_channels, in_channels, scale_size, self.num_funcs).float().to(device)/self.num_funcs)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # stdv = np.sqrt(1/(in_channels*kernel_size*kernel_size))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.bias.data.uniform_(-stdv, stdv)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
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
            return F.relu(output)#)self.norm(

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)
    
    
class Scale_SteerCNNs(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, num_layers, scales=[1.0], basis_type='A'):
        super(Scale_SteerCNNs, self).__init__()
        self.model = [SESConv_Z2_H(in_channels, hidden_dim, kernel_size, 7, scales=scales, padding=kernel_size // 2, bias=True, basis_type='A')]
        self.model += [SESConv_H_H(hidden_dim, hidden_dim, 1, kernel_size, 7, scales=scales, padding=kernel_size // 2, bias=True, basis_type='A') for i in range(num_layers - 2)]
        self.model += [SESConv_H_H(hidden_dim, out_channels, 1, kernel_size, 7, scales=[1.0], padding=kernel_size // 2, bias=True, basis_type='A', final_layer = True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x).squeeze(2)
    
    

###################################################    
############# Scale Group Convolution #############
################################################### 

class Scale_GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sout = 5):
        super(Scale_GroupConv2d, self).__init__()
        self.out_channels= out_channels
        self.in_channels = in_channels
        self.sout = sout
        self.kernel_size = kernel_size
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        weight_shape = (out_channels, 2, in_channels//2, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.Tensor(*weight_shape).to(device))
        self.stdv = math.sqrt(1. / (kernel_size * kernel_size * in_channels))
        self.reset_parameters()
        self.kernels = self.kernel_generation()
        #self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def reset_parameters(self):
        self.weights.data.uniform_(-self.stdv, self.stdv)
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
            #new_kernel = new_kernel/new_kernel.sum()*kernel.sum()
        return new_kernel[:,:,-kernel.shape[2]:]
    
    
    def kernel_generation(self):
        out = []
        for s in range(self.sout):
            if (s - self.sout//2) < 0:
                new_kernel = self.shrink_kernel(self.weights, (self.sout//2 - s)/2).to(device)
                
            # elif (s - self.sout//2) == 1:
            #     new_kernel = torch.einsum('cvtxy,txysij->cvsij', self.weights,
            #                               dilationdict[self.weights.shape[2]].to(self.weights.get_device())).to(device)
            elif (s - self.sout//2) > 0:
                new_kernel = self.dilate_kernel(self.weights, (s - self.sout//2)/2).to(device)
            else:
                new_kernel = self.weights.to(device)
            
            new_kernel = new_kernel.transpose(1,2)
            new_kernel = new_kernel.reshape(new_kernel.shape[0], -1, new_kernel.shape[3], new_kernel.shape[4])
            out.append(new_kernel.detach())
            
        return out
    
    def forward(self, xx, level):
        padding = ((self.kernels[level].shape[-2]-1)//2, (self.kernels[level].shape[-1]-1)//2)
        conv = F.conv2d(xx, self.kernels[level], padding = padding)
        out = F.relu(conv)
        return out, level
    
    
class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
    
class Scale_GroupConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, num_layers, num_scales):
        super(Scale_GroupConvNet, self).__init__()
        self.num_scales = num_scales
        
        layers = [Scale_GroupConv2d(in_channels, hidden_dim, kernel_size = kernel_size, sout = num_scales)]
        layers += [Scale_GroupConv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, sout = num_scales) for i in range(num_layers - 2)]
        #layers += [Scale_GroupConv2d(hidden_dim, out_channels, kernel_size = kernel_size, sout = num_scales)]
        self.model = mySequential(*layers)
        self.final_layer = nn.Conv2d(hidden_dim, out_channels, kernel_size = kernel_size, padding=(kernel_size-1)//2)
        
        
    def forward(self, xx):
        out = torch.stack([self.model(xx[l], l)[0] for l in range(self.num_scales)])
        out = self.final_layer(torch.mean(out, dim =0))
        return out