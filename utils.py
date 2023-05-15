import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils import data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Dataset(data.Dataset):
    def __init__(self, input_length, mid, output_length, direc, task_list, sample_list, stack = False):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc
        self.task_list = task_list
        self.sample_list = sample_list
        self.stack = stack
        try:
            self.data_lists = [torch.load(self.direc + "/raw_data_" + str(idx[0]) + "_" + str(idx[1]) + ".pt") for idx in task_list]
        except:
            self.data_lists = [torch.load(self.direc + "/raw_data_" + str(idx) + ".pt") for idx in task_list]

    def __len__(self):
        return len(self.task_list) * len(self.sample_list)

    def __getitem__(self, index):
        task_idx = index // len(self.sample_list)
        sample_idx = index % len(self.sample_list)        
        y = self.data_lists[task_idx][(self.sample_list[sample_idx]+self.mid):(self.sample_list[sample_idx]+self.mid+self.output_length)] 
        if not self.stack:
            x = self.data_lists[task_idx][(self.mid-self.input_length+self.sample_list[sample_idx]):(self.mid+self.sample_list[sample_idx])]
        else:
            x = self.data_lists[task_idx][(self.mid-self.input_length+self.sample_list[sample_idx]):(self.mid+self.sample_list[sample_idx])].reshape(-1, y.shape[-2], y.shape[-1])     
        return x.float(), y.float()
    
def train_epoch(train_loader, model, optimizer, loss_function):
    train_mse = []
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        loss = 0
        for y in yy.transpose(0,1):
            im = model(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += loss_function(im, y)  
        train_mse.append(loss.item()/yy.shape[1]) 
        try:
            weight_constraint = loss_function(model.module.get_weight_constraint(), torch.tensor(0).float().cuda())
            loss += weight_constraint
        except:
            pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_rmse = round(np.sqrt(np.mean(train_mse)),5)
    return train_rmse

def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            ims = []
            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, im.shape[1]:], im], 1)
                loss += loss_function(im, y)
                ims.append(im.cpu().data.numpy())
            ims = np.array(ims).transpose(1,0,2,3,4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item()/yy.shape[1])
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_rmse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_rmse, preds, trues


def RPPConv_L2(mdl, conv_wd = 1e-6, basic_wd = 1e-6):
    conv_l2 = 0.
    basic_l2 = 0.
    for block in mdl.model:
        if hasattr(block, 'conv'):
            conv_l2 += sum([p.pow(2).sum() for p in block.conv.parameters()])
        if hasattr(block, 'linear'):
            basic_l2 += sum([p.pow(2).sum() for p in block.linear.parameters()])
        
    return conv_wd*conv_l2  + basic_wd*basic_l2

def RPPConv_L1(mdl, conv_wd = 1e-6, basic_wd = 1e-6):
    conv_l1 = 0.
    basic_l1 = 0.
    for block in mdl.model:
        if hasattr(block, 'conv'):
            conv_l1 += sum([p.abs().sum() for p in block.conv.parameters()])
        if hasattr(block, 'linear'):
            basic_l1 += sum([p.abs().sum() for p in block.linear.parameters()])
        
    return conv_wd*conv_l1  + basic_wd*basic_l1