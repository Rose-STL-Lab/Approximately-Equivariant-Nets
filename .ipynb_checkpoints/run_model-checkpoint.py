import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils import data
from models import Relaxed_ConvNet, Relaxed_Rot_SteerConvNet, Relaxed_Scale_SteerCNNs
from utils import Dataset, train_epoch, eval_epoch, get_lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())


parser = argparse.ArgumentParser(description='Approximately Equivariant CNNs')
parser.add_argument('--dataset', type=str, required=False, default="PhiFlow", help='PhiFlow or JetFlow')
parser.add_argument('--hidden_dim', type=int, required=False, default="128", help='hidden dimension')
parser.add_argument('--num_layers', type=int, required=False, default="5", help='number of layers')
parser.add_argument('--out_length', type=int, required=False, default="6", help='number of prediction losses used for backpropagation')
parser.add_argument('--num_banks', type=int, required=False, default="2", help='number of filter banks used in relaxed group convolution')
parser.add_argument('--alpha', type=float, required=False, default="0", help='coefficient of the regularizer')
parser.add_argument('--input_length', type=int, required=False, default="10", help='input length')
parser.add_argument('--batch_size', type=int, required=False, default="8", help='batch size')
parser.add_argument('--num_epoch', type=int, required=False, default="1000", help='maximum number of epochs')
parser.add_argument('--learning_rate', type=float, required=False, default="0.001", help='learning rate')
parser.add_argument('--decay_rate', type=float, required=False, default="0.95", help='learning decay rate')
parser.add_argument('--relaxed_symmetry', type=str, required=True, default="Translation", help='translation or rotation or scaling')
args = parser.parse_args()


hidden_dim = args.hidden_dim
num_layers = args.num_layers
out_length = args.out_length
num_banks = args.num_banks
alpha = args.alpha
input_length = args.input_length
batch_size = args.batch_size
num_epoch = args.num_epoch
learning_rate = args.learning_rate
decay_rate = args.decay_rate
symmetry = args.relaxed_symmetry
mid = input_length + 2
min_rmse = 1e8




# Split time ranges
train_time = list(range(0, 160))
valid_time = list(range(160, 200))
test_future_time = list(range(200, 250))
test_domain_time = list(range(0, 100))

# Split tasks
if args.dataset == "PhiFlow":
    data_direc = args.dataset + "/" + symmetry
    h_size, w_size = 64, 64
    if symmetry == "Translation":
        train_task = [(48, 10), (56, 10), (8, 20),  (40, 5),  (56, 25), 
                      (48, 20), (48, 5),  (16, 20), (56, 5),  (32, 10), 
                      (56, 15), (16, 5),  (40, 15), (40, 25), (48, 25), 
                      (48, 15), (24, 10), (56, 20), (32, 15), (16, 15),
                      (8, 10),  (24, 15), (8, 15),  (32, 25), (8, 5)]

        test_domain_task = [(32, 20), (32, 5), (24, 20), (16, 25), (24, 5), 
                            (16, 10), (40, 20), (8, 25), (24, 25), (40, 10)]


    elif symmetry == "Rotation":
        train_task = [(27, 2), (33, 0), (3, 2), (28, 3),(9, 0),
                      (12, 3), (22, 1), (8, 3), (30, 1), (25, 0),
                      (16, 3), (11, 2), (23, 2), (29, 0), (36, 3),
                      (26, 1), (1, 0), (35, 2), (19, 2), (34, 1),
                      (4, 3), (2, 1), (7, 2), (31, 2), (17, 0)]

        test_domain_task = [(6, 1), (14, 1), (15, 2), (10, 1), (18, 1),
                            (20, 3), (24, 3), (13, 0), (21, 0), (5, 0)]

    elif symmetry == "Scale":
        train_task = [27,  9,  7, 11,  4, 26, 35,
                      2, 29, 10, 34, 12, 37, 28,
                      18, 24,  8, 14, 1, 31, 25,
                      0, 19, 15, 36,  3, 20, 13]

        test_domain_task = [ 5, 30, 16, 23, 33,
                             6, 17, 22, 21, 32]
        
elif args.dataset == "JetFlow":
    h_size, w_size = 62, 23
    data_direc = args.dataset
    train_task = [(1, 4), (3, 4), (2, 4), (1, 1), (2, 6), (3, 5),
                  (3, 3), (3, 1), (1, 8), (3, 8), (3, 6), (2, 1),
                  (1, 3), (1, 6), (2, 8), (1, 7), (1, 2), (2, 2)]

    test_domain_task = [(2, 3), (3, 7), (2, 7), (2, 5), (3, 2), (1, 5)]

    
else:
    print("Invalid dataset name entered!")

    
valid_task = train_task
test_future_task = train_task



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
                    task_list = valid_task,
                    sample_list = valid_time, 
                    stack = True)

test_set_future = Dataset(input_length = input_length, 
                          mid = mid, 
                          output_length = 20, 
                          direc = data_direc, 
                          task_list = test_future_task, 
                          sample_list = test_future_time, 
                          stack = True)

test_set_domain = Dataset(input_length = input_length, 
                          mid = mid, 
                          output_length = 20,
                          direc = data_direc, 
                          task_list = test_domain_task, 
                          sample_list = test_domain_time, 
                          stack = True)


train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = True, num_workers = 8)    
test_loader_future = data.DataLoader(test_set_future,  batch_size = batch_size, shuffle = False, num_workers = 8) 
test_loader_domain = data.DataLoader(test_set_domain,  batch_size = batch_size, shuffle = False, num_workers = 8)  



if symmetry == "Translation":
    
    model = nn.DataParallel(Relaxed_ConvNet(in_channels = input_length*2, 
                                            out_channels = 2,
                                            hidden_dim = hidden_dim, 
                                            kernel_size = 3, 
                                            h_size = h_size,
                                            w_size = w_size, 
                                            num_layers = num_layers, 
                                            num_banks = num_banks, 
                                            alpha = alpha).to(device))
    
    model_name = "Relaxed_ConvNet_bz{}_pred{}_lr{}_decay{}_hid{}_layers{}_banks{}_alpha{}".format(batch_size, 
                                                                                                  out_length, 
                                                                                                  learning_rate,
                                                                                                  decay_rate,
                                                                                                  hidden_dim,
                                                                                                  num_layers, 
                                                                                                  num_banks, 
                                                                                                  alpha)
elif symmetry == "Rotation":
    
    model = nn.DataParallel(Relaxed_Rot_SteerConvNet(in_frames = input_length, 
                                                     out_frames = 1, 
                                                     hidden_dim = hidden_dim,
                                                     kernel_size = 3, 
                                                     num_layers = num_layers,
                                                     N = 4, 
                                                     alpha = alpha).to(device))
    
    model_name = "Relaxed_Rot_SteerConvNet_bz{}_pred{}_lr{}_decay{}_hid{}_layers{}_alpha{}".format(batch_size, 
                                                                                                   out_length, 
                                                                                                   learning_rate, 
                                                                                                   decay_rate,
                                                                                                   hidden_dim,
                                                                                                   num_layers, 
                                                                                                   alpha)
  
elif symmetry == "Scale":
    
    model = nn.DataParallel(Relaxed_Scale_SteerCNNs(in_channels = input_length*2, 
                                                     out_channels = 2, 
                                                     hidden_dim = hidden_dim, 
                                                     kernel_size = 5, 
                                                     num_layers = num_layers, 
                                                     scales= [1.0,1.5,2.0,2.5],
                                                     basis_type='A',
                                                     alpha = alpha).to(device))
        
    model_name = "Relaxed_Scale_SteerCNNs_bz{}_pred{}_lr{}_decay{}_hid{}_layers{}_alpha{}".format(batch_size,
                                                                                                  out_length,
                                                                                                  learning_rate,
                                                                                                  decay_rate,
                                                                                                  hidden_dim, 
                                                                                                  num_layers, 
                                                                                                  alpha)
print(model_name)
print("number of paramters:", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)

optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=decay_rate)
loss_fun = torch.nn.MSELoss()

########################################################################################################################################################

train_rmse = []
valid_rmse = []

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
    if (len(train_rmse) > 100 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
            break       

    scheduler.step()
    print("Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(i+1, (end-start)/60, train_rmse[-1], valid_rmse[-1]))


future_rmse, future_preds, future_trues = eval_epoch(test_loader_future, best_model, loss_fun)
domain_rmse, domain_preds, domain_trues = eval_epoch(test_loader_domain, best_model, loss_fun)

print("Future Test RMSE:", future_rmse, ";", "Domain Test RMSE:", domain_rmse)
torch.save({"test_future": [future_rmse, future_preds, future_trues],
            "test_domain": [domain_rmse, domain_preds, domain_trues],
            "model": best_model.state_dict()}, 
             args.dataset + "_" + model_name + ".pt")

