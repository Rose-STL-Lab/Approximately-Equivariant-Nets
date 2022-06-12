import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### ConvNet ##########
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
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
    
######### MLP ##########
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
         
 
######### RPP ##########
class RPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_size, w_size, final_layer = False):
        super(RPPBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.linear = nn.Sequential(nn.Linear(in_channels*h_size*w_size, 32),
                                    nn.Linear(32, out_channels*h_size*w_size))
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.final_layer = final_layer
        
    def forward(self, x):
        convout = self.conv(x)
        linout = self.linear(x.view(x.shape[0], -1)).view(convout.shape)
        if self.final_layer:
            return convout + linout
        else:
            out = convout + linout#)
            return self.activation(self.bn(out))
    
class RPPNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_size, w_size, hidden_dim, num_layers):
        super(RPPNet, self).__init__()
        self.layers = [RPPBlock(in_channels, hidden_dim, kernel_size, h_size, w_size)]
        self.layers += [RPPBlock(hidden_dim, hidden_dim, kernel_size, h_size, w_size) for i in range(num_layers-2)]
        self.layers += [RPPBlock(hidden_dim, out_channels, kernel_size, h_size, w_size, final_layer = True)]
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.model(x)
    
    


######### Dynamic Filter ##########
class Dynamic_Filter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_size, w_size, mlp_hidden_dim, conv_hidden_dim):
        super(Dynamic_Filter, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*h_size*w_size, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden_dim, conv_hidden_dim*conv_hidden_dim*kernel_size*kernel_size + conv_hidden_dim)
        )
                
        self.conv_hidden_dim = conv_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.kernel_size = kernel_size 
        
        self.conv1 = ConvBlock(in_channels, conv_hidden_dim, kernel_size)
        self.conv2 = ConvBlock(conv_hidden_dim, conv_hidden_dim, kernel_size)
        self.conv4 = ConvBlock(conv_hidden_dim, conv_hidden_dim, kernel_size)
        self.conv5 = nn.Conv2d(conv_hidden_dim, out_channels, kernel_size, padding=(kernel_size-1)//2)
        
    def forward(self, x):
        encoder_out = self.encoder(x.view(x.shape[0], -1))
        weights = encoder_out[:,:-self.conv_hidden_dim]
        bias = encoder_out[:,-self.conv_hidden_dim:]
        weights = weights.reshape(x.shape[0], self.conv_hidden_dim, self.conv_hidden_dim, self.kernel_size, self.kernel_size)
        
        out = self.conv1(x)
        out = torch.cat([F.relu(F.conv2d(out[i:i+1], weights[i], bias = bias[i], 
                                                       padding = (self.kernel_size-1)//2)) for i in range(x.shape[0])], dim = 0)
        out = self.conv5(self.conv4(self.conv2(out)))
        return out
        
######### Lift Expansion ##########
class Lift_Expansion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_size, w_size, hidden_dim):
        super(Lift_Expansion, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*h_size*w_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size 
        
        self.conv1 = ConvBlock(in_channels, hidden_dim, kernel_size)
        self.conv2 = ConvBlock(hidden_dim, hidden_dim, kernel_size)
        self.conv3 = ConvBlock(hidden_dim*2, hidden_dim, kernel_size)
        self.conv4 = ConvBlock(hidden_dim, hidden_dim, kernel_size)
        self.conv5 = nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=(kernel_size-1)//2)
        
    def forward(self, x):
        encoder_out = self.encoder(x.view(x.shape[0], -1))
        encoder_out = encoder_out.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[-2], x.shape[-1])
        
        out = self.conv2(self.conv1(x))
        out = self.conv3(torch.cat([out, encoder_out], dim = 1))
        out = self.conv5(self.conv4(out))
        return out
    
######### MLP + Conv ##########
class MLPConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, num_mlp_layers, num_conv_layers, h_size, w_size):
        super(MLPConvNet, self).__init__()
        
        self.mlp_layers = [MLPBlock(in_channels*h_size*w_size, hidden_dim*2, h_size, w_size)]
        self.mlp_layers += [MLPBlock(hidden_dim*2, hidden_dim*2, h_size, w_size) for i in range(num_mlp_layers-2)]
        self.mlp_layers += [MLPBlock(hidden_dim*2, hidden_dim*h_size*w_size, h_size, w_size)]
        self.mlp = nn.Sequential(*self.mlp_layers)
        self.hidden_dim = hidden_dim
        
        self.conv_layers = [ConvBlock(hidden_dim, hidden_dim, kernel_size) for i in range(num_conv_layers-1)]
        self.conv_layers += [nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=(kernel_size-1)//2)]
        self.conv = nn.Sequential(*self.conv_layers)
        
    def forward(self, x):
        out = self.mlp(x).reshape(x.shape[0], self.hidden_dim, x.shape[2], x.shape[3])
        out = self.conv(out)
        return out
    
######### Relaxed Conv Net ##########    
class Relaxed_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_size, w_size, num_banks = 1, final_layer = False, norm = False):
        super(Relaxed_ConvBlock, self).__init__()
        self.convs = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2).to(device) for i in range(num_banks)])
        self.combination_weights = nn.Parameter(torch.ones(h_size, w_size, num_banks).float().to(device)/num_banks)
        # stdv = np.sqrt(1/num_banks)#*kernel_size*kernel_size
        # self.combination_weights.data.uniform_(-stdv, stdv)
        self.activation = nn.ReLU()
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size-1)//2
        self.h_size = h_size
        self.w_size = w_size
        self.final_layer = final_layer
        self.num_banks = num_banks
        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        outs = torch.stack([self.convs[i](x) for i in range(self.num_banks)], dim  = 0)
        
        # Compute Convolution
        out = torch.einsum("ijr, rboij -> boij", self.combination_weights, outs)
        if self.final_layer:
            return out
        else:
            if self.norm:
                return self.activation(self.bn(out))
            else:    
                return self.activation(out)
        
class Relaxed_ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, h_size, w_size, num_layers, num_banks = 1, alpha = 1, norm = False):
        super(Relaxed_ConvNet, self).__init__()
        layers = [Relaxed_ConvBlock(in_channels, hidden_dim, kernel_size, h_size, w_size, num_banks = num_banks, norm = norm)]
        layers += [Relaxed_ConvBlock(hidden_dim, hidden_dim, kernel_size, h_size, w_size, num_banks = num_banks, norm = norm) for i in range(num_layers-2)]
        layers += [Relaxed_ConvBlock(hidden_dim, out_channels, kernel_size, h_size, w_size, num_banks = num_banks, final_layer = True)]
        self.rconv = nn.Sequential(*layers)
        self.num_layers = num_layers
        self.alpha = alpha
        
    def get_weight_constraint(self):
        combine_weights = torch.cat([self.rconv[i].combination_weights.unsqueeze(-1) for i in range(self.num_layers)], dim = -1)
        return self.alpha * (torch.mean(torch.abs(combine_weights[1:] - combine_weights[:-1])) + torch.mean(torch.abs(combine_weights[:,1:] - combine_weights[:,:-1]))) 
    
    def get_mean_weights(self):
        return torch.mean(torch.abs(torch.cat([self.rconv[i].combination_weights.unsqueeze(-1) for i in range(self.num_layers)], dim = -1)))
          
    def forward(self, x):
        return self.rconv(x)
    
    

############ Constrained Locally Connected NN ############

class Constrained_LCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_size, w_size, final_layer = False):
        super(Constrained_LCBlock, self).__init__()
        stdv = np.sqrt(1/(in_channels*kernel_size*kernel_size))

        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, h_size, w_size).float().to(device)/in_channels)
        self.weights.data.uniform_(-stdv, stdv)

        self.bias = nn.Parameter(torch.randn(out_channels).float().to(device)/in_channels)
        self.bias.data.uniform_(-stdv, stdv)
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size-1)//2
        self.h_size = h_size
        self.w_size = w_size
        self.final_layer = final_layer
        
    def forward(self, x):
        x = F.unfold(x, kernel_size = self.kernel_size, padding = (self.kernel_size-1)//2)
        x = x.reshape(x.shape[0], self.in_channels, self.kernel_size, self.kernel_size, -1)
        x = x.reshape(x.shape[0], self.in_channels, self.kernel_size, self.kernel_size, self.h_size, self.w_size)
        
        out = torch.einsum("abcdij, rbcdij -> raij", self.weights, x)
        
        if self.final_layer:
            return out
        else:
            return self.activation(out)#)self.bn(
        
class Constrained_LCNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, h_size, w_size, num_layers, alpha = 1):
        super(Constrained_LCNet, self).__init__()
        layers = [Constrained_LCBlock(in_channels, hidden_dim, kernel_size, h_size, w_size)]
        layers += [Constrained_LCBlock(hidden_dim, hidden_dim, kernel_size, h_size, w_size) for i in range(num_layers-2)]
        layers += [Constrained_LCBlock(hidden_dim, out_channels, kernel_size, h_size, w_size, final_layer = True)]
        self.rconv = nn.Sequential(*layers)
        self.num_layers = num_layers
        self.alpha = alpha
        
    def spatial_diff(self, inp):
        return self.alpha * (torch.mean(torch.abs(inp[...,1:] - inp[...,:-1])) + torch.mean(torch.abs(inp[...,1:,:] - inp[...,:-1,:])))        
        
    def get_weight_constraint(self): 
        con = 0
        for i in range(self.num_layers):
            con += self.spatial_diff(self.rconv[i].weights) 
        return con
          
    def forward(self, x):
        return self.rconv(x)
    

        