import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset
from torchvision import transforms
import os 
import time
from tqdm import trange

d_PGM_ViT = torch.load('data/PGM_shape_size_color_normalized.pt').flatten(start_dim=3) # [7, 10, 10, 1600]
def load_PGM_inputs_ViT(attr): 
    """attr: (3, 9, 3), (num_panel, num_pos, num_attr)"""
    img_seq =  -0.6891*torch.ones((3*9, 40*40))
    for i_panel in range(3): 
        for i_pos in range(9): 
            if attr[i_panel, i_pos, 0] != -1: 
                i_shape, i_size, i_color = attr[i_panel, i_pos]
                img_seq[i_panel*9+i_pos] = d_PGM_ViT[int(i_shape), int(i_size), int(i_color)]
    return img_seq

class dataset_PGM_single_ViT(Dataset): 
    def __init__(self, attr_list): 
        self.attr_list = attr_list  
        
    def __len__(self): 
        return len(self.attr_list)
    
    def __getitem__(self, idx): 
        attr = self.attr_list[idx] # torch.Size([3, 9, 3])
        inputs = load_PGM_inputs_ViT(attr)
        return inputs

class dataset_PGM_ViT(Dataset): 
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
        inputs = load_PGM_inputs_ViT(attr)
        return inputs, target

import os 
def setup_savename(save_name): 
    fas_dir = '/n/netscratch/sompolinsky_lab/Lab/shane/Code_theory/'
    save_dir = fas_dir+save_name+'/'
    log_file = 'log/'+save_name+'.txt'

    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    if not os.path.exists(save_dir+'param/'): 
        os.mkdir(save_dir+'param/')
    if not os.path.exists(log_file): 
        with open(log_file, 'w') as f: 
            f.write('Start!\n')
    return save_dir, log_file

from functools import partial
from torchvision.models.vision_transformer import Encoder
class Shane_ViT(nn.Module): 
    def __init__(self, num_layers, num_heads, hidden_dim, mlp_dim, seq_length=27, 
                 inputs_dim=1600, num_classes=35, dropout=0.0, attention_dropout=0.0, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)): 
        super().__init__()

        self.proj = nn.Linear(inputs_dim, hidden_dim)
        
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = Encoder(seq_length+1, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
        self.seq_length = seq_length

        self.heads = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.heads.weight)
        nn.init.zeros_(self.heads.bias)
    
    def forward(self, x):
        """x: (1, 27, 1600), (batch_size, seq_length, hidden_dim) """
        # Conv2d layer 
        x = self.proj(x)
        batch_size = x.shape[0]
        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1) # torch.Size([1, 28, 1600])

        # step 2: encoder block 
        x = self.encoder(x) # torch.Size([1, 28, 1600])

        feature = x[:, 0] # torch.Size([1, 1600])
        outputs = self.heads(feature) # torch.Size([1, 35])
        return outputs, feature

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

def train_epoch(model, loader_train, optimizer, device, warmup_lr_epoch): 
    start_time = time.time()
    model.train()
    running_loss = 0.0 
    correct, total = 0.0, 0.0 

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
        pred = outputs.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum().numpy()
        total += target.size()[0]
        
        del inputs, target, outputs, feature, loss 

    loss = running_loss/len(loader_train)
    acc = 100.0 * correct / total 
    dur = time.time()-start_time
        
    return acc, loss, dur 
    
from torch.utils.data import Dataset, DataLoader
def load_data_ViT(gen_id, batch_size_train, device):
    
    attr_train_all = torch.load('data/attr_train.pt') # torch.Size([40, 10000, 3, 9, 3])
    attr_val_all = torch.load('data/attr_val.pt')
    attr_test_all = torch.load('data/attr_test.pt')

    train_id = [i for i in np.arange(40) if i not in gen_id]
    
    train_inputs = attr_train_all[train_id] # torch.Size([35, 10000, 3, 9, 3])
    val_inputs = attr_val_all[train_id]
    gen_inputs = attr_test_all[gen_id]

    dataset_train = dataset_PGM_ViT(train_inputs)
    loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True, pin_memory=True)

    dataset_val_list = [dataset_PGM_single_ViT(x) for x in val_inputs]
    loader_val_list = [DataLoader(d, batch_size=500, shuffle=False, pin_memory=True) for d in dataset_val_list]

    dataset_gen_list = [dataset_PGM_single_ViT(x) for x in gen_inputs]
    loader_gen_list = [DataLoader(d, batch_size=500, shuffle=False, pin_memory=True) for d in dataset_gen_list]

    return loader_train, loader_val_list, loader_gen_list