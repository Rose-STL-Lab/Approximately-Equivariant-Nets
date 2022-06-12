import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils import data
from models import Relaxed_ConvNet, Relaxed_Rot_SteerConvNet, Relaxed_Scale_SteerCNNs
from train_utils import Dataset, train_epoch, eval_epoch, get_lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

########################################################################################################################################################

hidden_dim = 128
num_layers = 5
out_length = 6
num_banks = 2
alpha = 0
input_length = 10
mid = input_length + 2
batch_size = 8
num_epoch = 1000
learning_rate = 0.001
min_mse = 10

########################################################################################################################################################

train_task = [(48, 10), (56, 10), (8, 20),  (40, 5),  (56, 25), 
              (48, 20), (48, 5),  (16, 20), (56, 5),  (32, 10), 
              (56, 15), (16, 5),  (40, 15), (40, 25), (48, 25), 
              (48, 15), (24, 10), (56, 20), (32, 15), (16, 15),
              (8, 10),  (24, 15), (8, 15),  (32, 25), (8, 5)]

train_time = list(range(0, 160))

valid_task = train_task
valid_time = list(range(160, 200))

test_future_task = train_task
test_future_time = list(range(200, 250))

test_domain_task = [(32, 20), (32, 5), (24, 20), (16, 25), (24, 5), 
                    (16, 10), (40, 20), (8, 25), (24, 25), (40, 10)]
test_domain_time = list(range(0, 100))

direc = "Data/Translation"

########################################################################################################################################################

train_set = Dataset(input_length = input_length, mid = mid, output_length = out_length, direc = direc, 
            task_list = train_task, sample_list = train_time, stack = True)

valid_set = Dataset(input_length = input_length, mid = mid, output_length = out_length, direc = direc, 
                    task_list = valid_task, sample_list = valid_time, stack = True)

test_set_future = Dataset(input_length = input_length, mid = mid, output_length = 20, direc = direc, 
                          task_list = test_future_task, sample_list = test_future_time, stack = True)

test_set_domain = Dataset(input_length = input_length, mid = mid, output_length = 20, direc = direc, 
                          task_list = test_domain_task, sample_list = test_domain_time, stack = True)


train_loader = data.DataLoader(train_set,  batch_size = batch_size, shuffle = True, num_workers = 8)

valid_loader = data.DataLoader(valid_set,  batch_size = batch_size, shuffle = True, num_workers = 8)    

# two test sets
test_loader_future = data.DataLoader(test_set_future,  batch_size = batch_size, shuffle = False, num_workers = 8) 

test_loader_domain = data.DataLoader(test_set_domain,  batch_size = batch_size, shuffle = False, num_workers = 8)  

########################################################################################################################################################

name = "Relaxed_ConvNet_bz{}_pred{}_lr{}_hid{}_layers{}_banks{}_alpha{}".format(batch_size, out_length, learning_rate, 
                                                                                hidden_dim, num_layers, num_banks, alpha)
last_epoch = 0
try:
    model, last_epoch, learning_rate = torch.load(name + ".pth")
    print("Resume Training")
    print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
except:
    model = nn.DataParallel(Relaxed_ConvNet(in_channels = input_length*2, out_channels = 2, hidden_dim = hidden_dim, kernel_size = 3, 
                                            h_size = 64, w_size = 64, num_layers = num_layers, num_banks = num_banks, alpha = alpha).to(device))
    print("New model")
print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)

optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.9)
loss_fun = torch.nn.MSELoss()

########################################################################################################################################################

train_mse = []
valid_mse = []
test_mse = []

for i in range(last_epoch, num_epoch):
    start = time.time()

    model.train()
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun))

    model.eval()
    mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save([model, i, get_lr(optimizer)], name + ".pth")

    end = time.time()
    
    # Early Stopping
    if (len(train_mse) > 100 - last_epoch and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break       

    scheduler.step()
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)

########################################################################################################################################################

best_model = torch.load(name + ".pth")[0]
loss_fun = torch.nn.MSELoss()
torch.save({"test_future": eval_epoch(test_loader_future, best_model, loss_fun),
            "test_domain": eval_epoch(test_loader_domain, best_model, loss_fun)}, 
             name + ".pt")


########################################################################################################################################################


# hidden_dim = 128
# num_layers = 5
# out_length = 6
# alpha = 1e-6
# input_length = 10
# mid = input_length + 2
# batch_size = 16
# num_epoch = 100
# learning_rate = 0.001
# min_mse = 1

# train_task = [(27, 2), (33, 0), (3, 2), (28, 3),(9, 0),
#               (12, 3), (22, 1), (8, 3), (30, 1), (25, 0),
#               (16, 3), (11, 2), (23, 2), (29, 0), (36, 3),
#               (26, 1), (1, 0), (35, 2), (19, 2), (34, 1),
#               (4, 3), (2, 1), (7, 2), (31, 2), (17, 0)]

# train_time = list(range(0, 160))

# valid_task = train_task
# valid_time = list(range(160, 200))

# test_future_task = train_task
# test_future_time = list(range(200, 250))


# test_domain_task = [(6, 1), (14, 1), (15, 2), (10, 1), (18, 1),
#                     (20, 3), (24, 3), (13, 0), (21, 0), (5, 0)]

# test_domain_time = list(range(0, 100))

# direc = "Data/Rotation"

# train_set = Dataset(input_length = input_length, mid = mid, output_length = out_length, direc = direc, 
#             task_list = train_task, sample_list = train_time, stack = True)

# valid_set = Dataset(input_length = input_length, mid = mid, output_length = out_length, direc = direc, 
#                     task_list = valid_task, sample_list = valid_time, stack = True)

# test_set_future = Dataset(input_length = input_length, mid = mid, output_length = 20, direc = direc, 
#                           task_list = test_future_task, sample_list = test_future_time, stack = True)

# test_set_domain = Dataset(input_length = input_length, mid = mid, output_length = 20, direc = direc, 
#                           task_list = test_domain_task, sample_list = test_domain_time, stack = True)


# train_loader = data.DataLoader(train_set,  batch_size = batch_size, shuffle = True, num_workers = 8)

# valid_loader = data.DataLoader(valid_set,  batch_size = batch_size, shuffle = True, num_workers = 8)    

# test_loader_future = data.DataLoader(test_set_future,  batch_size = batch_size, shuffle = False, num_workers = 8) 

# test_loader_domain = data.DataLoader(test_set_domain,  batch_size = batch_size, shuffle = False, num_workers = 8)  

# name = "Relaxed_Rot_SteerConvNet_bz{}_pred{}_lr{}_hid{}_layers{}_alpha{}3".format(batch_size, out_length, learning_rate, hidden_dim, num_layers, alpha)
# try:
#     saved_model, last_epoch, learning_rate = torch.load(name + ".pth")
#     model = nn.DataParallel(Relaxed_Rot_SteerConvNet(in_frames = input_length, out_frames = 1, hidden_dim = hidden_dim,
#                                                  kernel_size = kernel_size, num_layers = num_layers, N = 4, alpha = alpha).to(device))
#     model.load_state_dict(saved_model)
#     print("Resume Training")
#     print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
# except:
#     model = nn.DataParallel(Relaxed_Rot_SteerConvNet(in_frames = input_length, out_frames = 1, hidden_dim = hidden_dim,
#                                                      kernel_size = 3, num_layers = num_layers, N = 4, alpha = alpha).to(device))
#     last_epoch = 0
#     print("New model")
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)

# optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.9)
# loss_fun = torch.nn.MSELoss()

# train_mse = []
# valid_mse = []
# test_mse = []

# for i in range(last_epoch, num_epoch):
#     start = time.time()
#     scheduler.step()

#     model.train()
#     train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun))

#     model.eval()
#     mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
#     valid_mse.append(mse)

#     if valid_mse[-1] < min_mse:
#         min_mse = valid_mse[-1] 
#         best_model = model
#         torch.save([model.state_dict(), i, get_lr(optimizer)], name + ".pth")

#     end = time.time()
#     if (len(train_mse) > 100 - last_epoch and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
#             break       
#     print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)


# loss_fun = torch.nn.MSELoss()
# torch.save({"test_future": eval_epoch(test_loader_future, best_model, loss_fun),
#             "test_domain": eval_epoch(test_loader_domain, best_model, loss_fun)}, 
#              name + ".pt")

########################################################################################################################################################


# out_length = 6
# hidden_dim = 128
# num_layers = 5
# alpha = 1e-6
# learning_rate = 0.0001
# input_length = 10
# mid = input_length + 2
# batch_size = 8
# num_epoch = 1000
# min_mse = 1

# train_task = [27,  9,  7, 11,  4, 26, 35,
#               2, 29, 10, 34, 12, 37, 28,
#               18, 24,  8, 14, 1, 31, 25,
#               0, 19, 15, 36,  3, 20, 13]


# train_time = list(range(0, 160))

# valid_task = train_task
# valid_time = list(range(160, 200))

# test_future_task = train_task
# test_future_time = list(range(200, 250))


# test_domain_task = [ 5, 30, 16, 23, 33,
#                      6, 17, 22, 21, 32]

# test_domain_time = list(range(0, 100))

# direc = "Data/Scale"

# ########################################################################################################################################################

# train_set = Dataset(input_length = input_length, mid = mid, output_length = out_length, direc = direc, 
#             task_list = train_task, sample_list = train_time, stack = True)

# valid_set = Dataset(input_length = input_length, mid = mid, output_length = out_length, direc = direc, 
#                     task_list = valid_task, sample_list = valid_time, stack = True)

# test_set_future = Dataset(input_length = input_length, mid = mid, output_length = 20, direc = direc, 
#                           task_list = test_future_task, sample_list = test_future_time, stack = True)

# test_set_domain = Dataset(input_length = input_length, mid = mid, output_length = 20, direc = direc, 
#                           task_list = test_domain_task, sample_list = test_domain_time, stack = True)


# train_loader = data.DataLoader(train_set,  batch_size = batch_size, shuffle = True, num_workers = 8)

# valid_loader = data.DataLoader(valid_set,  batch_size = batch_size, shuffle = True, num_workers = 8)    

# test_loader_future = data.DataLoader(test_set_future,  batch_size = batch_size, shuffle = False, num_workers = 8) 

# test_loader_domain = data.DataLoader(test_set_domain,  batch_size = batch_size, shuffle = False, num_workers = 8)  

# ########################################################################################################################################################

# name = "Relaxed_Scale_SteerCNNs_bz{}_pred{}_lr{}_hid{}_layers{}_alpha{}".format(batch_size, out_length, learning_rate, hidden_dim, num_layers, alpha)
# try:
#     saved_model, last_epoch, learning_rate = torch.load(name + ".pth")
#     model = nn.DataParallel(Relaxed_Scale_SteerCNNs(in_channels = input_length*2, out_channels = 2, hidden_dim = hidden_dim, kernel_size = 5, 
#                                     num_layers = num_layers, scales= [1.0,1.5,2.0,2.5], basis_type='A', alpha = alpha).to(device))
#     model.load_state_dict(saved_model)

#     print("Resume Training:", name)
#     print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
# except:
#     model = nn.DataParallel(Relaxed_Scale_SteerCNNs(in_channels = input_length*2, out_channels = 2, hidden_dim = hidden_dim, kernel_size = 5, 
#                                             num_layers = num_layers, scales= [1.0,1.5,2.0,2.5], basis_type='A', alpha = alpha).to(device))
#     last_epoch = 0
#     print("New model:", name)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)


# optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
# loss_fun = torch.nn.MSELoss()

# train_mse = []
# valid_mse = []
# test_mse = []

# for i in range(last_epoch, num_epoch):

#     start = time.time()
#     scheduler.step()

#     model.train()
#     train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun))

#     model.eval()
#     mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
#     valid_mse.append(mse)

#     if valid_mse[-1] < min_mse:
#         min_mse = valid_mse[-1] 
#         best_model = model
#         #torch.save([model, i, get_lr(optimizer)], name + ".pth")
#         torch.save([model.state_dict(), i, get_lr(optimizer)], name + ".pth")

#     end = time.time()
#     if (len(train_mse) > 80 - last_epoch and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
#             break       
#     print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)


# best_model = nn.DataParallel(Relaxed_Scale_SteerCNNs(in_channels = input_length*2, out_channels = 2, hidden_dim = hidden_dim, kernel_size = 5, 
#                                                      num_layers = num_layers, scales= [1.0,1.5,2.0,2.5], basis_type='A', alpha = alpha).to(device))
# best_model.load_state_dict(torch.load(name + ".pth")[0])

# loss_fun = torch.nn.MSELoss()
# torch.save({"test_future": eval_epoch(test_loader_future, best_model, loss_fun),
#             "test_domain": eval_epoch(test_loader_domain, best_model, loss_fun)}, 
#              name + ".pt")

