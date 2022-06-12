import torch
import e2cnn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from e2cnn.nn.modules.r2_conv.r2convolution import compute_basis_params
from e2cnn.nn.modules.r2_conv.basisexpansion_singleblock import block_basisexpansion
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
    
    

######################################################  
############ Rotationally Equivariant CNN ############
######################################################


class E2Conv(torch.nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, N):
        super(E2Conv, self).__init__()
        
        r2_act = e2cnn.gspaces.Rot2dOnR2(N = N)
        feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.regular_repr])
        feat_type_hid = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.regular_repr])
        
        self.layer = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(feat_type_in, feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            e2cnn.nn.InnerBatchNorm(feat_type_hid),
            e2cnn.nn.ReLU(feat_type_hid)
        ) 
        
    def forward(self, xx):
        return self.layer(xx)

class E2CNN(torch.nn.Module):
    def __init__(self, in_frames, out_frames, hidden_dim, kernel_size, num_layers, N):
        super(E2CNN, self).__init__()
        r2_act = e2cnn.gspaces.Rot2dOnR2(N = N)
        
        self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.irrep(1)])
        self.feat_type_hid = e2cnn.nn.FieldType(r2_act, hidden_dim*[r2_act.regular_repr])
        self.feat_type_out = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.irrep(1)])
        
        input_layer = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            e2cnn.nn.InnerBatchNorm(self.feat_type_hid),
            e2cnn.nn.ReLU(self.feat_type_hid)
        ) 
        
        layers = [input_layer]
        layers += [E2Conv(hidden_dim, hidden_dim, kernel_size, N) for i in range(num_layers-2)]
        layers += [e2cnn.nn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size = kernel_size, padding = (kernel_size - 1)//2)]
        self.model = torch.nn.Sequential(*layers)
    
        
    def forward(self, xx):
        xx = e2cnn.nn.GeometricTensor(xx, self.feat_type_in)
        out = self.model(xx)
        return out.tensor
    
    
########################################  
############ Lift_Expansion ############
########################################

class Lift_Rot_Expansion(torch.nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, encoder_hidden_dim, backbone_hidden_dim, N):
        super(Lift_Rot_Expansion, self).__init__() 
        self.encoder = nn.Sequential(
            nn.Linear(in_frames*2*62*23, encoder_hidden_dim),
            nn.BatchNorm1d(encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            nn.BatchNorm1d(encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, backbone_hidden_dim*N//4)
        )

        r2_act = e2cnn.gspaces.Rot2dOnR2(N = N)
        self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.irrep(1)])
        self.feat_type_hid = e2cnn.nn.FieldType(r2_act, backbone_hidden_dim*[r2_act.regular_repr]) 
        self.feat_type_hid_2 = e2cnn.nn.FieldType(r2_act, (backbone_hidden_dim + backbone_hidden_dim//4)*[r2_act.regular_repr]) 
        self.feat_type_out = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.irrep(1)])
        
        self.e2conv_1 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            e2cnn.nn.InnerBatchNorm(self.feat_type_hid),
            e2cnn.nn.ReLU(self.feat_type_hid),
            e2cnn.nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            e2cnn.nn.InnerBatchNorm(self.feat_type_hid),
            e2cnn.nn.ReLU(self.feat_type_hid)
        ) 
        
        self.e2conv_2 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(self.feat_type_hid_2, self.feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            e2cnn.nn.InnerBatchNorm(self.feat_type_hid),
            e2cnn.nn.ReLU(self.feat_type_hid),
            e2cnn.nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            e2cnn.nn.InnerBatchNorm(self.feat_type_hid),
            e2cnn.nn.ReLU(self.feat_type_hid),
            e2cnn.nn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size = kernel_size, padding = (kernel_size - 1)//2)
        ) 
        
    def forward(self, x):
        encoder_out = self.encoder(x.reshape(x.shape[0], -1))
        encoder_out = encoder_out.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[-2], x.shape[-1])

        out = e2cnn.nn.GeometricTensor(x, self.feat_type_in)
        out = self.e2conv_1(out)
        out = torch.cat([out.tensor, encoder_out], dim = 1)
        out = e2cnn.nn.GeometricTensor(out, self.feat_type_hid_2)
        out = self.e2conv_2(out)
        return out.tensor
    
    
    
#########################################  
############ ConvNet + E2CNN ############
#########################################

class ConvE2CNN(torch.nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, conv_hidden_dim, e2cnn_hidden_dim, conv_num_layers, e2cnn_num_layers,  N):
        super(ConvE2CNN, self).__init__()
        
        self.convnet = [ConvBlock(in_frames*2, conv_hidden_dim, kernel_size)]
        self.convnet += [ConvBlock(conv_hidden_dim, conv_hidden_dim, kernel_size) for i in range(conv_num_layers-2)]
        self.convnet += [ConvBlock(conv_hidden_dim, e2cnn_hidden_dim*N, kernel_size)]
        self.convnet = nn.Sequential(*self.convnet)
        
        r2_act = e2cnn.gspaces.Rot2dOnR2(N = N)
        self.feat_type_hid = e2cnn.nn.FieldType(r2_act, e2cnn_hidden_dim*[r2_act.regular_repr])
        self.feat_type_out = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.irrep(1)])
        
        self.e2cnn = [E2Conv(e2cnn_hidden_dim, e2cnn_hidden_dim, kernel_size, N) for i in range(e2cnn_num_layers-2)]
        self.e2cnn += [e2cnn.nn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size = kernel_size, padding = (kernel_size - 1)//2)]
        self.e2cnn = torch.nn.Sequential(*self.e2cnn)
    
        
    def forward(self, xx):
        out = self.convnet(xx)
        out = e2cnn.nn.GeometricTensor(out, self.feat_type_hid)
        out = self.e2cnn(out)
        return out.tensor
    

    
#####################################################
############ Rotational Residual Pathway ############
#####################################################  

class RPPBlock(nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, N, first_layer = False, final_layer = False):
        super(RPPBlock, self).__init__()
        r2_act = e2cnn.gspaces.Rot2dOnR2(N = N)
        self.first_layer = first_layer
        self.final_layer = final_layer
        
        # E2 Equivariant Layer
        if self.first_layer:
            self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.irrep(1)])
        else:
            self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.regular_repr])
            
        if self.final_layer:
            self.feat_type_hid = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.irrep(1)])
        else:
            self.feat_type_hid = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.regular_repr])
        self.e2cnn = e2cnn.nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2)
        
        # Regular Convolution Layer
        if self.first_layer:
            self.conv = nn.Conv2d(in_frames*2, out_frames*N, kernel_size, padding=(kernel_size-1)//2)
        elif self.final_layer:
            self.conv = nn.Conv2d(in_frames*N, out_frames*2, kernel_size, padding=(kernel_size-1)//2)
        else:
            self.conv = nn.Conv2d(in_frames*N, out_frames*N, kernel_size, padding=(kernel_size-1)//2)

        self.norm = nn.BatchNorm2d(out_frames*N)
        self.activation = nn.ReLU()
        
        
    def forward(self, x):
        convout = self.conv(x)
        e2cnnout = self.e2cnn(e2cnn.nn.GeometricTensor(x, self.feat_type_in)).tensor
        if self.final_layer:
            return convout + e2cnnout
        else:
            out = convout + e2cnnout
            return self.activation(self.norm(out))
    
class Rot_RPPNet(nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, N, hidden_dim, num_layers):
        super(Rot_RPPNet, self).__init__()
        self.layers = [RPPBlock(in_frames = in_frames, out_frames = hidden_dim, kernel_size = kernel_size, 
                                N = N, first_layer = True, final_layer = False)]
        self.layers += [RPPBlock(in_frames = hidden_dim, out_frames = hidden_dim, kernel_size = kernel_size, 
                                 N = N, first_layer = False, final_layer = False) for i in range(num_layers-2)]
        self.layers += [RPPBlock(in_frames = hidden_dim, out_frames = out_frames, kernel_size = kernel_size, 
                                 N = N, first_layer = False, final_layer = True)]
        self.model = nn.Sequential(*self.layers)
        
        
    def get_weight_constraint(self, conv_wd = 1e-6):
        conv_l2 = 0.
        basic_l2 = 0.
        for block in self.model:
            if hasattr(block, 'conv'):
                conv_l2 += sum([p.pow(2).sum() for p in block.conv.parameters()])
        return conv_wd*conv_l2
        
    def forward(self, x):
        return self.model(x)
    



##########################################################    
############ Constrained Locally Connected NN ############
##########################################################

class Constrained_LCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_size, w_size, final_layer = False):
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
        
    def rot_vector(self, inp, theta):
        #inp shape: c x 2 x 64 x 64
        theta = torch.tensor(theta).float().to(device)
        rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float().to(device)
        out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
        return out
    
    def get_rot_mat(self, theta):
        theta = torch.tensor(theta).float().to(device)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]]).float().to(device)

    def rot_img(self, x, theta):
        rot_mat = self.get_rot_mat(theta)[None, ...].float().repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).float()
        x = F.grid_sample(x, grid)
        return x.float()
    
    def get_rotated_kernels(self, theta):
        temp_w = self.weights.reshape(self.h_size, self.w_size, self.out_channels, self.in_channels//2, 2, self.kernel_size, self.kernel_size)#.clone()
        temp_w = temp_w.reshape(-1, 2, self.kernel_size, self.kernel_size)
        temp_w = self.rot_vector(temp_w, theta)
        temp_w = temp_w.reshape(-1, 2, self.kernel_size, self.kernel_size)
        temp_w = torch.cat([self.rot_img(temp_w[:,:1], theta), self.rot_img(temp_w[:,1:2], theta)], dim = 1)
        temp_w = temp_w.reshape(self.h_size, self.w_size, self.out_channels//2, 2, self.in_channels, self.kernel_size, self.kernel_size)
        temp_w = temp_w.reshape(-1, 2, self.in_channels, self.kernel_size, self.kernel_size)
        temp_w = self.rot_vector(temp_w, theta)
        temp_w = temp_w.reshape(self.h_size, self.w_size, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return temp_w
        
        
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
        
class Constrained_Rot_LCNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, h_size, w_size, num_layers, alpha = 1, N = 4):
        super(Constrained_Rot_LCNet, self).__init__()
        layers = [Constrained_LCBlock(in_channels, hidden_dim, kernel_size, h_size, w_size)]
        layers += [Constrained_LCBlock(hidden_dim, hidden_dim, kernel_size, h_size, w_size) for i in range(num_layers-2)]
        layers += [Constrained_LCBlock(hidden_dim, out_channels, kernel_size, h_size, w_size, final_layer = True)]
        self.rconv = nn.Sequential(*layers)
        self.num_layers = num_layers
        self.alpha = alpha
        self.N = N
        
    def get_weight_constraint(self): 
        constraint = 0
        theta_step = np.pi*2/self.N
        for layer in self.rconv:
            for j in range(1, self.N):
                constraint += F.mse_loss(layer.weights, layer.get_rotated_kernels(theta_step*j))
                #print(F.mse_loss(layer.weights, layer.get_rotated_kernels(theta_step*j)).item())
        return self.alpha*constraint
          
    def forward(self, x):
        return self.rconv(x)
    
    
###################################################    
############ Relaxed Group Convolution ############
################################################### 
class Relaxed_Reg_GroupConv(nn.Module):
    def __init__(self, in_reps, out_reps, kernel_size, N, num_filter_banks, first_layer = False, final_layer = False):
        super(Relaxed_Reg_GroupConv, self).__init__()
        self.num_filter_banks = num_filter_banks
        self.N = N
        
        self.first_layer = first_layer
        self.final_layer = final_layer

        self.kernel_size = kernel_size
        self.out_reps = out_reps
        self.in_reps = in_reps
        
        stdv = np.sqrt(1/(self.in_reps))
        
        ## If this is first layer, rho_1 -> rho_reg
        if self.first_layer:
            self.transform_weights = nn.Parameter(torch.ones(in_reps, 1).float().to(device)/2)
            #self.transform_weights.data.uniform_(-stdv, stdv)
            
        if self.final_layer == True:
            self.final_w = nn.Parameter(torch.randn(out_reps).to(device))
            self.final_w.data.uniform_(-stdv, stdv)

        ## Initialize weights
        self.combination_weights = nn.Parameter(torch.ones(self.N, self.num_filter_banks).float().to(device)/self.num_filter_banks)
        self.combination_weights.data.uniform_(-stdv, stdv)
        
        self.weights = nn.Parameter(torch.randn(self.num_filter_banks, out_reps, self.N, in_reps, self.N, kernel_size, kernel_size).to(device))
        self.weights.data.uniform_(-stdv, stdv)

        self.bias = nn.Parameter(torch.zeros(out_reps, self.N).to(device))
        self.bias.data.uniform_(-stdv, stdv)
        
        self.batch_norm = nn.BatchNorm2d(out_reps*self.N)
        
    def permute(self, weights, bias):
        augmented_weights = []
        augmented_bias = []
        
        for i in range(self.N):
            permuted_indices = list(np.roll(np.arange(0, self.N, 1), shift = i))

            temp_w = weights[i, :, :, :, permuted_indices,...][:, permuted_indices]
            temp_w = temp_w.reshape(self.out_reps*self.N, self.in_reps, self.N, self.kernel_size, self.kernel_size) 
            temp_w = temp_w.reshape(self.out_reps*self.N, self.in_reps*self.N, self.kernel_size, self.kernel_size) 
            temp_b = bias[:, permuted_indices]
            
            augmented_weights.append(temp_w)
            augmented_bias.append(temp_b.reshape(-1))
            
        return torch.cat(augmented_weights, dim = 0), torch.cat(augmented_bias, dim = 0)
    
    def rot_vector(self, inp, theta):
        #inp shape: c x 2 x 64 x 64
        theta = torch.tensor(theta).float().to(device)
        rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float().to(device)
        out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
        return out

    def rot_img(self, x, theta):
        theta = torch.tensor(theta).float().to(device)
        get_rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]]).float().to(device)
        rot_mat = get_rot_mat[None, ...].float().repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).float()
        x = F.grid_sample(x, grid)
        return x.float()
 
    def forward(self, x):
        
        if self.first_layer:
            xs = []
            x = x.reshape(x.shape[0], x.shape[1]//2, 2, x.shape[-2], x.shape[-1])
            theta = torch.tensor(2*np.pi/self.N).float()
            for i in range(self.N):
                lift = torch.tensor([torch.cos(theta*i), torch.sin(theta*i)]).float().to(device)
                lift_weights = torch.einsum("ab, b -> ab", self.transform_weights.repeat(1,2), lift)
                xs.append(torch.einsum("abcde, bc -> abde", x, lift_weights).unsqueeze(2))
            xs = torch.cat(xs, dim = 2)
            xs = xs.reshape(xs.shape[0], xs.shape[1]*xs.shape[2], xs.shape[3], xs.shape[4])
        else:
            xs = x
        
        conv_weights = torch.einsum("na, ab... -> nb...", self.combination_weights.to(self.weights.device), self.weights)
        augmented_weights, augmented_biases = self.permute(conv_weights, self.bias) 
        
        out = F.conv2d(xs, augmented_weights, augmented_biases, padding = (self.kernel_size - 1)//2)
        out = out.reshape(out.shape[0], self.N, self.out_reps*self.N, out.shape[-2], out.shape[-1]).mean(1)
        if self.final_layer == True:
            theta = torch.tensor(2*np.pi/self.N).float()
            out = out.reshape(out.shape[0], self.out_reps, self.N, out.shape[-2], out.shape[-1])
            out_u = torch.sum(torch.stack([out[:,:,i:i+1]*torch.cos(theta*i) for i in range(self.N)]), dim = 0)
            out_v = torch.sum(torch.stack([out[:,:,i:i+1]*torch.sin(theta*i) for i in range(self.N)]), dim = 0)
            out = torch.cat([out_u, out_v], dim  = 2)
            out = torch.einsum("abcde, b -> abcde", out, self.final_w)
            out = out.reshape(out.shape[0], self.out_reps*2, out.shape[-2], out.shape[-1])
            return out
        else:
            return F.relu(out)#)self.batch_norm(

class Relaxed_Reg_GroupConvNet(nn.Module):
    def __init__(self, in_reps, out_reps, hidden_dim, kernel_size, num_layers, num_filter_banks, N):
        super(Relaxed_Reg_GroupConvNet, self).__init__()     
        
        layers = [Relaxed_Reg_GroupConv(in_reps = in_reps, out_reps = hidden_dim, kernel_size = kernel_size, 
                                    N = N, num_filter_banks = num_filter_banks, first_layer = True, final_layer = False)]
        
        layers += [Relaxed_Reg_GroupConv(in_reps = hidden_dim, out_reps = hidden_dim, kernel_size = kernel_size, 
                                     N = N, num_filter_banks = num_filter_banks, first_layer = False, final_layer = False) for i in range(num_layers-2)]
        
        layers += [Relaxed_Reg_GroupConv(in_reps = hidden_dim, out_reps = out_reps, kernel_size = kernel_size, 
                                     N = N, num_filter_banks = num_filter_banks, first_layer = False, final_layer = True)]
        
        self.rconv = nn.Sequential(*layers)
    def rot_vector(self, inp, theta):
        #inp shape: c x 2 x 64 x 64
        theta = torch.tensor(theta).float().to(device)
        rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float().to(device)
        out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
        return out

    def rot_img(self, x, theta):
        theta = torch.tensor(theta).float().to(device)
        get_rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]]).float().to(device)
        rot_mat = get_rot_mat[None, ...].float().repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).float()
        x = F.grid_sample(x, grid)
        return x.float()
                  
    def forward(self, x):
        return self.rconv(x)
    


    
#######################################################    
############ Relaxed Steerable Convolution ############
#######################################################

class Relaxed_Rot_SteerConv(torch.nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, N, first_layer = False, last_layer = False):
        super(Relaxed_Rot_SteerConv, self).__init__()
        r2_act = e2cnn.gspaces.Rot2dOnR2(N = N)
        self.last_layer = last_layer
        self.first_layer = first_layer
        self.kernel_size = kernel_size
        
        if self.first_layer:
            self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.irrep(1)])
        else:
            self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.regular_repr])
            
        if self.last_layer:
            self.feat_type_hid = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.irrep(1)])
        else:
            self.feat_type_hid = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.regular_repr])
            
        if not last_layer:
            self.norm = e2cnn.nn.InnerBatchNorm(self.feat_type_hid)
            self.relu = e2cnn.nn.ReLU(self.feat_type_hid)
        
        
        grid, basis_filter, rings, sigma, maximum_frequency = compute_basis_params(kernel_size = kernel_size)
        i_repr = self.feat_type_in._unique_representations.pop()
        o_repr = self.feat_type_hid._unique_representations.pop()
        basis = self.feat_type_in.gspace.build_kernel_basis(i_repr, o_repr, sigma, rings, maximum_frequency = 5)
        block_expansion = block_basisexpansion(basis, grid, basis_filter, recompute=False)

        
        self.basis_kernels = block_expansion.sampled_basis.to(device)
        
        
        stdv = np.sqrt(1/(in_frames*kernel_size*kernel_size))
        self.relaxed_weights = nn.Parameter(torch.ones(out_frames, self.basis_kernels.shape[0], in_frames, kernel_size**2).float().to(device))
        self.relaxed_weights.data.uniform_(-stdv, stdv)

        self.bias = nn.Parameter(torch.zeros(out_frames*self.basis_kernels.shape[1]).to(device))
        self.bias.data.uniform_(-stdv, stdv)
        
        # self.relaxed_weights = nn.Parameter(torch.ones(out_frames, self.basis_kernels.shape[0], in_frames, kernel_size**2).float().to(device)/self.basis_kernels.shape[0]/(kernel_size**2))
        # self.bias = nn.Parameter(torch.zeros(out_frames*self.basis_kernels.shape[1]).to(device))

        
    def get_weight_constraint(self):
        return torch.mean(torch.abs(self.relaxed_weights[...,:-1] - self.relaxed_weights[...,1:])) #torch.roll()

    def forward(self, x):
        conv_filters = torch.einsum('bpqk,obik->opiqk', self.basis_kernels.to(self.relaxed_weights.device), self.relaxed_weights) 
        conv_filters = conv_filters.reshape(conv_filters.shape[0]*conv_filters.shape[1],
                                            conv_filters.shape[2]*conv_filters.shape[3], 
                                            self.kernel_size, self.kernel_size)
        
        if not self.last_layer:
            out = F.conv2d(x, conv_filters, self.bias, padding = 1)
            return self.relu(e2cnn.nn.GeometricTensor(out, self.feat_type_hid)).tensor#self.norm(
        else:
            return F.conv2d(x, conv_filters, self.bias, padding = 1)
        
class Relaxed_Rot_SteerConvNet(torch.nn.Module):
    def __init__(self, in_frames, out_frames, hidden_dim, kernel_size, num_layers, N, alpha = 1):
        super(Relaxed_Rot_SteerConvNet, self).__init__()
        self.alpha = alpha

        layers = [Relaxed_Rot_SteerConv(in_frames = in_frames, out_frames = hidden_dim, 
                                 kernel_size = kernel_size, N = N, 
                                 first_layer = True, last_layer = False)]
        
        layers += [Relaxed_Rot_SteerConv(in_frames = hidden_dim, out_frames = hidden_dim, 
                                  kernel_size = kernel_size, N = N, 
                                  first_layer = False, last_layer = False) 
                   for i in range(num_layers-2)]
        
        layers += [Relaxed_Rot_SteerConv(in_frames = hidden_dim, out_frames = out_frames, 
                                  kernel_size = kernel_size, N = N, 
                                  first_layer = False, last_layer = True) ]
        self.model = torch.nn.Sequential(*layers)
        
    def rot_vector(self, inp, theta):
        #inp shape: c x 2 x 64 x 64
        theta = torch.tensor(theta).float().to(device)
        rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float().to(device)
        out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
        return out
    
    def get_rot_mat(self, theta):
        theta = torch.tensor(theta).float().to(device)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]]).float().to(device)

    def rot_img(self, x, theta):
        rot_mat = self.get_rot_mat(theta)[None, ...].float().repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).float()
        x = F.grid_sample(x, grid)
        return x.float()
   
    def get_weight_constraint(self):
        return self.alpha * sum([layer.get_weight_constraint() for layer in self.model])
    
        
    def forward(self, xx):
        return self.model(xx)
    

################################################################################   
############ Relaxed Rotation and Translation Steerable Convolution ############
################################################################################
class Relaxed_R_SteerConv(torch.nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, N, first_layer = False, last_layer = False):
        super(Relaxed_R_SteerConv, self).__init__()
        r2_act = e2cnn.gspaces.Rot2dOnR2(N = N)
        self.last_layer = last_layer
        self.first_layer = first_layer
        self.kernel_size = kernel_size
        
        if self.first_layer:
            self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.irrep(1)]) 
        else:
            self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames*[r2_act.regular_repr])
            
        if self.last_layer:
            self.feat_type_hid = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.irrep(1)])
        else:
            self.feat_type_hid = e2cnn.nn.FieldType(r2_act, out_frames*[r2_act.regular_repr]) 
            
        if not last_layer:
            self.norm = e2cnn.nn.InnerBatchNorm(self.feat_type_hid)
            self.relu = e2cnn.nn.ReLU(self.feat_type_hid)
        
        
        grid, basis_filter, rings, sigma, maximum_frequency = compute_basis_params(kernel_size = kernel_size)
        i_repr = self.feat_type_in._unique_representations.pop()
        o_repr = self.feat_type_hid._unique_representations.pop()
        basis = self.feat_type_in.gspace.build_kernel_basis(i_repr, o_repr, sigma, rings, maximum_frequency = 5)
        block_expansion = block_basisexpansion(basis, grid, basis_filter, recompute=False)

        
        self.basis_kernels = block_expansion.sampled_basis.to(device)  
        
        
        stdv = np.sqrt(1/(in_frames*kernel_size*kernel_size))
        self.relaxed_weights = nn.Parameter(torch.ones(out_frames, self.basis_kernels.shape[0], in_frames, kernel_size**2).float().to(device))
        self.relaxed_weights.data.uniform_(-stdv, stdv)

        self.bias = nn.Parameter(torch.zeros(out_frames*self.basis_kernels.shape[1]).to(device))
        self.bias.data.uniform_(-stdv, stdv)
        
    def get_weight_constraint(self):
        return torch.mean(torch.abs(self.relaxed_weights[...,:-1] - self.relaxed_weights[...,1:])) #torch.roll()

    def forward(self, x):
        conv_filters = torch.einsum('bpqk,obik->opiqk', self.basis_kernels.to(self.relaxed_weights.device), self.relaxed_weights) 
        conv_filters = conv_filters.reshape(conv_filters.shape[0]*conv_filters.shape[1],
                                            conv_filters.shape[2]*conv_filters.shape[3], 
                                            self.kernel_size, self.kernel_size)
        
        return F.conv2d(x, conv_filters, self.bias, padding = 1)
        
class Relaxed_TR_SteerConv(nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, N, num_banks, h_size, w_size, first_layer = False, last_layer = False):
        super(Relaxed_TR_SteerConv, self).__init__()
        self.convs = nn.Sequential(*[Relaxed_R_SteerConv(in_frames = in_frames, out_frames = out_frames, 
                                                       kernel_size = kernel_size, N = N, first_layer = first_layer, 
                                                       last_layer = last_layer).to(device) for i in range(num_banks)])
        
        self.combination_weights = nn.Parameter(torch.ones(h_size, w_size, num_banks).float().to(device)/num_banks)
        
        #self.activation = nn.ReLU()
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size-1)//2
        self.h_size = h_size
        self.w_size = w_size
        self.last_layer = last_layer
        self.num_banks = num_banks
            

    def get_weight_constraint(self):
        return sum([layer.get_weight_constraint() for layer in self.convs])
        
    def forward(self, x):
        outs = torch.stack([self.convs[i](x) for i in range(self.num_banks)], dim  = 0)
        
        # Compute Convolution
        out = torch.einsum("ijr, rboij -> boij", self.combination_weights, outs)
        
        
        if self.last_layer:
            return out
        else:
            return self.convs[0].relu(e2cnn.nn.GeometricTensor(out, self.convs[0].feat_type_hid)).tensor
        
        
class Relaxed_TR_SteerConvNet(torch.nn.Module):
    def __init__(self, in_frames, out_frames, hidden_dim, kernel_size, num_layers, N, num_banks, h_size, w_size, alpha = 1):
        super(Relaxed_TR_SteerConvNet, self).__init__()
        self.alpha = alpha

        layers = [Relaxed_TR_SteerConv(in_frames = in_frames, out_frames = hidden_dim, 
                                       kernel_size = kernel_size, N = N, num_banks = num_banks,
                                       h_size = h_size, w_size = w_size, first_layer = True, last_layer = False)]
        
        layers += [Relaxed_TR_SteerConv(in_frames = hidden_dim, out_frames = hidden_dim, 
                                        kernel_size = kernel_size, N = N, num_banks = num_banks,
                                        h_size = h_size, w_size = w_size, first_layer = False, last_layer = False) 
                   for i in range(num_layers-2)]
        
        layers += [Relaxed_TR_SteerConv(in_frames = hidden_dim, out_frames = out_frames, 
                                        kernel_size = kernel_size, N = N, num_banks = num_banks,
                                        h_size = h_size, w_size = w_size, first_layer = False, last_layer = True) ]
        self.model = torch.nn.Sequential(*layers)
        
    def rot_vector(self, inp, theta):
        #inp shape: c x 2 x 64 x 64
        theta = torch.tensor(theta).float().to(device)
        rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float().to(device)
        out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
        return out
    
    def get_rot_mat(self, theta):
        theta = torch.tensor(theta).float().to(device)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]]).float().to(device)

    def rot_img(self, x, theta):
        rot_mat = self.get_rot_mat(theta)[None, ...].float().repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).float()
        x = F.grid_sample(x, grid)
        return x.float()
   
    def get_weight_constraint(self):
        return self.alpha * sum([layer.get_weight_constraint() for layer in self.model])
    
        
    def forward(self, xx):
        return self.model(xx)
    
