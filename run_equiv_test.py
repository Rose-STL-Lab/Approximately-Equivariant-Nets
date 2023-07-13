import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import random
from torch.utils import data
from models import E2CNN, Relaxed_Rot_SteerConvNet, ConvNet, Rot_RPPNet, Lift_Rot_Expansion
from utils import Dataset, train_epoch, eval_epoch, get_lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Data Equiv Errors: | {}".format([0.0, 0.17, 0.313, 0.435, 0.541, 0.63, 0.709, 0.78, 0.84, 0.896]))

hidden_dim = 64
num_layers = 5
out_length = 6
alpha = 0 
input_length = 1
batch_size = 32
num_epoch = 1000
learning_rate = 0.001
decay_rate = 0.9
mid = input_length + 2
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Split time ranges
train_time = list(range(0, 30))
valid_time = list(range(30, 40))

def rot_vector(inp, theta):
    #inp shape: c x 2 x 64 x 64
    theta = torch.tensor(theta).float().to(device)
    rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float().to(device)
    out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
    return out

def get_rot_mat(theta):
    theta = torch.tensor(theta).float().to(device)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]]).float().to(device)

def rot_img(x, theta):
    rot_mat = get_rot_mat(theta)[None, ...].float().repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).float()
    x = F.grid_sample(x, grid)
    return x.float()

def rot_field(x, theta):
    x_rot = torch.cat([rot_img(rot_vector(x, theta)[:,:1],  theta), 
                       rot_img(rot_vector(x, theta)[:,-1:], theta)], dim = 1)
    return x_rot


for model_name in ["E2CNN", "ConvNet", "RPP", "Lift", "RSteer"]:#
    equiv_error_lst = []
    for level in range(10):
        min_rmse = 1e8
        data_direc = "equivariance_test/E_" + str(level)
        train_task = [0, 1, 2, 3]

        train_set = Dataset(input_length = input_length, 
                            mid = mid, 
                            output_length = out_length,
                            direc = data_direc, 
                            task_list = train_task, 
                            sample_list = train_time, 
                            stack = True)

        valid_set = Dataset(input_length = input_length, 
                            mid = mid, 
                            output_length = out_length, 
                            direc = data_direc, 
                            task_list = train_task,
                            sample_list = valid_time, 
                            stack = True)

        train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
        valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 2) 
        test_loader = data.DataLoader(valid_set, batch_size = 1, shuffle = False, num_workers = 2)    


        if model_name == "E2CNN":
            model = nn.DataParallel(E2CNN(in_frames= input_length, 
                                          out_frames = 1, 
                                          hidden_dim = hidden_dim,
                                          kernel_size = 3, 
                                          num_layers = num_layers,
                                          N = 4).to(device))

        elif model_name == "RSteer":
            model = nn.DataParallel(Relaxed_Rot_SteerConvNet(in_frames = input_length, 
                                                             out_frames = 1, 
                                                             hidden_dim = hidden_dim//2, 
                                                             kernel_size = 3, 
                                                             num_layers = num_layers, 
                                                             N = 4, 
                                                             alpha = alpha).to(device))

        elif model_name == "ConvNet":
            model = nn.DataParallel(ConvNet(in_channels = input_length*2, 
                                            out_channels = 2,
                                            hidden_dim = hidden_dim,
                                            kernel_size = 3, 
                                            num_layers = num_layers).to(device))

        elif model_name == "RPP":
            model = nn.DataParallel(Rot_RPPNet(in_frames = input_length,
                                               out_frames = 1,
                                               hidden_dim = hidden_dim,
                                               kernel_size = 3, 
                                               num_layers = num_layers,
                                               N = 4).to(device))
        elif model_name == "Lift":
            model = nn.DataParallel(Lift_Rot_Expansion(in_frames = input_length,
                                                       out_frames = 1, 
                                                       kernel_size = 3, 
                                                       encoder_hidden_dim = hidden_dim//2, 
                                                       backbone_hidden_dim = hidden_dim//2, 
                                                       N = 4).to(device))



        optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=decay_rate)
        loss_fun = torch.nn.MSELoss()

        ########################################################################################################################################################

        train_rmse = []
        valid_rmse = []
        best_model = 0

        for i in range(num_epoch):
            start = time.time()

            model.train()
            train_rmse.append(train_epoch(train_loader, model, optimizer, loss_fun))

            model.eval()
            mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
            valid_rmse.append(mse)

            if valid_rmse[-1] < min_rmse:
                min_rmse = valid_rmse[-1] 
                best_model = model

            end = time.time()

            # Early Stopping
            if (len(train_rmse) > 50 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
                    break       

            scheduler.step()


        equiv_errors = []
        with torch.no_grad():
            for xx, yy in test_loader:
                xx = xx.to(device)
                orig_pred = best_model(xx).reshape(-1, 2, xx.shape[-2], xx.shape[-1])

                for angle in [np.pi/2, np.pi, np.pi/2*3]:
                    rho_inp = rot_field(xx.reshape(-1, 2, xx.shape[-2], xx.shape[-1]), angle).to(device)
                    rho_inp = rho_inp.reshape(1, -1, xx.shape[-2], xx.shape[-1])
                    rho_inp_outs = best_model(rho_inp).reshape(-1, 2, xx.shape[-2], xx.shape[-1])
                    equiv_errors.append(torch.mean(torch.abs(rho_inp_outs - rot_field(orig_pred, angle))).data.cpu())
        equiv_error_lst.append(np.round(np.mean(equiv_errors),3))

    print("{} Equiv Errors: | {}".format(model_name, equiv_error_lst))
    



