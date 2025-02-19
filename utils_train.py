import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

import os

gen_id_list = [
    [1, 16, 20, 34, 37], 
    [8, 12, 24, 36, 39],
    [5, 17, 21, 33, 38],
    [3, 10, 29, 31, 37],
    [0, 14, 27, 35, 38],
    [4, 19, 26, 30, 39],
    [9, 13, 25, 32, 37],
    [2, 18, 23, 30, 38],
    [7, 15, 22, 34, 39],
    [6, 11, 28, 33, 37],
]

import os 
def setup_savename(save_name): 
    fas_dir = '/n/netscratch/sompolinsky_lab/Lab/shane/Code_theory/'
    save_dir = fas_dir+save_name+'/'
    log_file = 'log/'+save_name+'.txt'

    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    if not os.path.exists(log_file): 
        with open(log_file, 'w') as f: 
            f.write('Start!\n')
    return save_dir, log_file

from torch.utils.data import Dataset, DataLoader
def load_data(gen_id, P=None, sampler_train=None):
    attr_train_all = torch.load('data/attr_train.pt', weights_only=True) # torch.Size([40, 10000, 3, 9, 3])
    attr_test_all = torch.load('data/attr_test.pt', weights_only=True)

    if P is not None: 
        attr_train_all = attr_train_all[:,:P]

    train_id = [i for i in range(40) if i not in gen_id]

    train_inputs = attr_train_all[train_id] # torch.Size([35, 10000, 3, 9, 3])
    val_inputs = attr_test_all[train_id]
    gen_inputs = attr_test_all[gen_id]

    dataset_train = dataset_PGM(train_inputs)

    if sampler_train is not None: 
        loader_train = DataLoader(dataset_train, shuffle=False, pin_memory=True, batch_sampler=sampler_train)
    else:
        loader_train = DataLoader(dataset_train, batch_size=512, shuffle=True, pin_memory=True)

    dataset_val_list = [dataset_PGM_single(x) for x in val_inputs]
    loader_val_list = [DataLoader(d, batch_size=500, shuffle=False, pin_memory=True) for d in dataset_val_list]

    dataset_gen_list = [dataset_PGM_single(x) for x in gen_inputs]
    loader_gen_list = [DataLoader(d, batch_size=500, shuffle=False, pin_memory=True) for d in dataset_gen_list]

    if sampler_train is not None: 
        return loader_train, loader_val_list, loader_gen_list, sampler_train
    else: 
        return loader_train, loader_val_list, loader_gen_list

from torch.utils.data import Sampler
class Sampler_class(Sampler): 
    """always sample in classes"""
    def __init__(self, num_batches, M_batch, P_class_batch, M, P_class): 
        
        self.seed, self.num_batches = 0, num_batches
        self.M_batch, self.P_class_batch = M_batch, P_class_batch
        self.M, self.P_class = M, P_class
        
    def __iter__(self): 
        np.random.seed(self.seed)
        for i in range(self.num_batches): 
            classes_train = np.random.choice(self.M, self.M_batch, replace=False)
            indices = []
            for i_class in classes_train: 
                ind_list = np.random.choice(self.P_class, self.P_class_batch, replace=False)
                indices.extend(i_class*self.P_class+ind_list)
            yield indices
    
    def __len__(self): 
        return self.num_batches
    
    def set_epoch(self, epoch): 
        self.seed = epoch
        
def load_data_batch(gen_id, num_batches, M_batch, P_class_batch):
    
    attr_train_all = torch.load('data/attr_train.pt') # torch.Size([40, 10000, 3, 9, 3])
    attr_val_all = torch.load('data/attr_val.pt')
    attr_test_all = torch.load('data/attr_test.pt')

    train_id = [i for i in range(40) if i not in gen_id]

    train_inputs = attr_train_all[train_id] # torch.Size([35, 10000, 3, 9, 3])
    val_inputs = attr_val_all[train_id]
    gen_inputs = attr_test_all[gen_id]
    
    dataset_train = dataset_PGM(train_inputs)
    M, P_class = train_inputs.shape[:2] # 35, num_samples_total
    sampler_train = Sampler_class(num_batches, M_batch, P_class_batch, M, P_class)
    loader_train = DataLoader(dataset_train, shuffle=False, pin_memory=True, batch_sampler=sampler_train)

    dataset_val_list = [dataset_PGM_single(x) for x in val_inputs]
    loader_val_list = [DataLoader(d, batch_size=500, shuffle=False, pin_memory=True) for d in dataset_val_list]

    dataset_gen_list = [dataset_PGM_single(x) for x in gen_inputs]
    loader_gen_list = [DataLoader(d, batch_size=500, shuffle=False, pin_memory=True) for d in dataset_gen_list]
    
    return loader_train, loader_val_list, loader_gen_list, sampler_train

#################### PGM ####################
pos_list = [[20,20], [20,60], [20,100], [60,20], [60,60], [60,100], [100,20], [100,60], [100,100]]
d_PGM = torch.load('data/20240308_data/PGM_shape_size_color_normalized.pt', weights_only=True) # torch.Size([7, 10, 10, 40, 40])

def load_PGM_inputs(attr): 
    """attr: (3, 9, 3), (num_panel, num_pos, num_attr)"""
    inputs = -0.6891*torch.ones((3, 160, 160))
    for i_panel in range(3): 
        for i_pos in range(9): 
            if attr[i_panel, i_pos, 0] != -1: 
                i_shape, i_size, i_color = attr[i_panel, i_pos]
                x0, y0 = pos_list[i_pos]
                inputs[i_panel, x0:(x0+40), y0:(y0+40)] = d_PGM[int(i_shape), int(i_size), int(i_color)]
    return inputs 

class dataset_PGM_single(Dataset): 
    def __init__(self, attr_list): 
        self.attr_list = attr_list  
        
    def __len__(self): 
        return len(self.attr_list)
    
    def __getitem__(self, idx): 
        attr = self.attr_list[idx] # torch.Size([3, 9, 3])
        inputs = load_PGM_inputs(attr)
        return inputs
    
class dataset_PGM(Dataset): 
    def __init__(self, attr_all): 
        """inputs_all: # [38, 10000, 3, 9, 3]"""
        num_classes, num_samples = attr_all.shape[:2]
        target_all = torch.arange(num_classes)
        target_all = target_all.unsqueeze(1).repeat(1, num_samples)
        
        self.attr_all = attr_all.flatten(end_dim=1)  
        self.target_all = target_all.flatten(end_dim=1) 
        
    def __len__(self): 
        return len(self.target_all)
    
    def __getitem__(self, idx): 
        attr = self.attr_all[idx]
        target = self.target_all[idx]
        inputs = load_PGM_inputs(attr)
        return inputs, target

def train_epoch(model, loader_train, optimizer, device, warmup_lr_epoch): 
    
    model.train()
    running_loss = 0.0 

    for i_batch, (inputs, target) in enumerate(loader_train): 
        
        if warmup_lr_epoch is not None: 
            for p in optimizer.param_groups:
                p['lr'] = warmup_lr_epoch[i_batch]
        
        inputs = inputs.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        outputs, feature = model(inputs) 
        
        loss = nn.CrossEntropyLoss()(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        del inputs, target, outputs, feature, loss 
        
    return running_loss/len(loader_train)

def get_feature(model, loader_list, device, get_acc=True): 
    model.eval()
    feature_all = []
    correct, total = 0.0, 0.0 
    running_loss_CE = 0.0 
    
    with torch.no_grad(): 
        for i_class, loader in enumerate(loader_list): 
            feature_class = []
            for i_batch, inputs in enumerate(loader): 
                inputs = inputs.to(device, non_blocking=True).float()
                outputs, feature = model(inputs)
                feature_class.extend(feature)
                
                if get_acc: 
                    target = i_class * torch.ones(len(inputs))
                    target = target.to(device, non_blocking=True).long()
                    pred = outputs.data.max(1)[1]
                    correct += pred.eq(target.data).cpu().sum().numpy()
                    total += target.size()[0]

                    loss_CE = nn.CrossEntropyLoss()(outputs, target)
                    running_loss_CE += loss_CE.item()
                    
                del inputs, outputs, feature 
            feature_all.append(torch.stack(feature_class, dim=0))
    feature_all = torch.stack(feature_all)
    
    if get_acc: 
        acc = 100.0 * correct / total 
        loss_CE = running_loss_CE / len(loader_list)
    else: 
        acc, loss_CE = None, None 

    return feature_all, acc, loss_CE

### SNR 
from torch.autograd import Variable
import numpy as np
def get_SNR_matrix(feature, device):
    """
    feature: (num_classes, num_samples, dim_feature)
    
    checked that 10x faster than previous version
    """
    # step 1: get center 
    num_classes, num_samples, dim_feature = feature.shape 
    D = min([num_samples, dim_feature])
    x0 = feature.mean(1) # (num_classes, dim_feature)
    
    # step 2: get Ri, ui 
    Ri_list = [] # (num_classes, dim_feature)
    U_list = [] # (num_classes, dim_feature, num_samples)
    for i_class in range(num_classes): 
        _, R_svd, U = torch.linalg.svd(feature[i_class]-x0[i_class], full_matrices=False)
        Ri = np.sqrt(D/num_samples) * R_svd
        Ri_list.append(Ri)
        U_list.append(U.T @ torch.diag(Ri))
    Ri_list = torch.stack(Ri_list) # torch.Size([3, 40])
    U_list = torch.stack(U_list) # torch.Size([3, 80, 40])
    
    # step 3: Ra2, Da 
    Ra2_list = torch.mean(Ri_list**2, axis=1) # torch.Size([3]), (num_classes,)
    Da_list = D * torch.div(Ra2_list**2, torch.mean(Ri_list**4, axis=1)) # torch.Size([3]), (num_classes,)
    Ra_list = torch.sqrt(Ra2_list) # torch.Size([3]), (num_classes,)
    
    # step 4: get signal 
    a = x0.expand(num_classes, num_classes, dim_feature) # torch.Size([3, 3, 80])
    diff = a - a.transpose(0,1) # torch.Size([3, 3, 80])
    # (i,j): class_i - class_j 
    
    dx = diff.flatten(end_dim=1)[1:].reshape(num_classes-1,num_classes+1, dim_feature)[:,:-1].reshape(num_classes, num_classes-1, dim_feature)
    dx0 = dx / torch.sqrt(Ra2_list).reshape(num_classes, 1, 1) # torch.Size([3, 2, 80]), # normalize 

    # term 1: signal (**checked, 2023.05.26) 
    signal = (dx0**2).sum(2) # torch.Size([3, 2])
    
    # term 2: bias (**checked, 2023.05.26) 
    a1 = Ra2_list.unsqueeze(1).expand(num_classes, num_classes)
    Ra2_ratio = a1.transpose(0,1) / a1 # (3, 3)
    Ra2_ratio = Ra2_ratio.flatten()[1:].reshape(num_classes-1,num_classes+1)[:,:-1].reshape(num_classes, num_classes-1) # (3,2)
    bias = Ra2_ratio - 1.0 # (3,2)
    
    # term 3: vdim (**checked, 2023.05.26) 
    vdim = (1.0/Da_list).unsqueeze(1).expand(num_classes, num_classes-1)
    
    # term 4: vbias (**checked, 2023.05.26) 
    a2 = (D/(D+2))*(1.0/Da_list - 1.0/D)* (Ra2_list**2) # torch.Size([3])
    a3 = a2.unsqueeze(1).expand(num_classes, num_classes) + a2.unsqueeze(0).expand(num_classes, num_classes) # torch.Size([3, 3])
    a4 = a3.flatten()[1:].reshape(num_classes-1,num_classes+1)[:,:-1].reshape(num_classes, num_classes-1)
    vbias = torch.div(a4, (Ra2_list**2).unsqueeze(1))/2 # torch.Size([3, 2])
    
    # term 5: signoise (**checked, 2023.05.26) 
    indices = torch.arange(num_classes).unsqueeze(0).expand(num_classes, num_classes)
    indices = indices.flatten()[1:].reshape(num_classes-1,num_classes+1)[:,:-1].reshape(num_classes, num_classes-1)
    indices = Variable(indices, requires_grad=False).to(device)
    
    Ub = torch.index_select(U_list, 0, indices.flatten().long()) # torch.Size([6, 80, 40])

    Ua = U_list.unsqueeze(1).expand(num_classes, num_classes-1, dim_feature, D).flatten(end_dim=1) # torch.Size([6, 80, 40])

    U_ab = torch.stack([Ua, Ub], dim=1) # torch.Size([6, 2, 80, 40])
    dx0_flatten = dx0.reshape(num_classes*(num_classes-1), 1, 1, dim_feature) # torch.Size([6, 1, 1, 80])
    signoise = torch.matmul(dx0_flatten, U_ab).squeeze(2)  # torch.Size([6, 2, 40])
    signoise = (signoise**2).sum(2).sum(1).reshape(num_classes, num_classes-1) # torch.Size([3, 2])
    signoise = torch.div(signoise, (D*Ra2_list).unsqueeze(1)) # torch.Size([3, 2])
    
    # term 6: nnoise (**checked, 2023.05.26)
    nnoise = (torch.bmm(Ua.transpose(1,2), Ub)**2).flatten(start_dim=1).sum(1).reshape(num_classes, num_classes-1)
    nnoise = torch.div(nnoise, ((D*Ra2_list)**2).unsqueeze(1)) # torch.Size([3, 2])
    
    # sum the terms (**checked, 2023.05.26)
    SNR = torch.div( (signal+bias)/2 , torch.sqrt(vdim + vbias + signoise + nnoise) )
    
    return SNR, signal, bias, vdim, vbias, signoise, nnoise, Da_list

def get_SNR_matrix_numpy(feature, device): 
    SNR_stats = get_SNR_matrix(feature, device)
    return [x.detach().cpu().numpy() for x in SNR_stats]