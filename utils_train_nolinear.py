import torch 
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch.nn as nn
import os 
import numpy as np 

class Sampler_class_full(Sampler): 
    """always sample in classes, each batch contains all classes"""
    def __init__(self, num_batches, num_samples_per_class, 
                        num_classes_total=35, num_samples_total=10000): 
        
        self.seed = 0
        self.num_batches = num_batches 
        self.num_samples_per_class = num_samples_per_class 
        self.num_classes_total = num_classes_total
        self.num_samples_total = num_samples_total
        
    def __iter__(self): 
        np.random.seed(self.seed)
        for i in range(self.num_batches): 
            indices = []
            for i_class in range(self.num_classes_total): 
                ind_list = np.random.choice(self.num_samples_total, self.num_samples_per_class, replace=False)
                indices.extend(i_class*self.num_samples_total+ind_list)
            yield indices
    
    def __len__(self): 
        return self.num_batches
    
    def set_epoch(self, epoch): 
        self.seed = epoch

def train_epoch_nonlinear(model, loss_fn, loader_train, optimizer, device, warmup_lr_epoch): 
    
    model.train()
    running_loss = 0.0 

    for i_batch, (inputs, target) in enumerate(loader_train): 
        
        if warmup_lr_epoch is not None: 
            for p in optimizer.param_groups:
                p['lr'] = warmup_lr_epoch[i_batch]
        
        inputs = inputs.to(device, non_blocking=True).float() # [560, 3, 160, 160]
        target = target.to(device, non_blocking=True).long() # [560]

        optimizer.zero_grad(set_to_none=True)
        feature = model(inputs) # [560, 400]
        
        loss = loss_fn(feature)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        del inputs, target, feature, loss 
        
    return running_loss/len(loader_train)

from utils_train import get_SNR_matrix
def loss_fn_SNR_nolinear(feature, device, num_classes_per_batch, num_samples_per_class, temperature=1.0): 
    """SNR loss only"""
    feature = feature.reshape(num_classes_per_batch, num_samples_per_class, -1)
    SNR = get_SNR_matrix(feature, device)[0]
    return torch.exp(-SNR/temperature).mean()

from torch.autograd import Variable
def loss_fn_Proto(feature, device, num_classes_per_batch, num_samples_per_class, dim_feature=400, k_shot=1):
    # step 1: reshape feature 
    feature = feature.reshape(num_classes_per_batch, num_samples_per_class, dim_feature) # torch.Size([35, 16, 400])
    num_query = num_samples_per_class-k_shot
    
    # step 2: separate feature to proto and query 
    feature_proto = feature[:,:k_shot].mean(1).unsqueeze(0).expand(num_classes_per_batch*num_query, num_classes_per_batch, dim_feature) # torch.Size([525, 35, 400])
    feature_query = feature[:,k_shot:].flatten(end_dim=1).unsqueeze(1).expand(num_classes_per_batch*num_query, num_classes_per_batch, dim_feature) # torch.Size([525, 35, 400])
    logits = -((feature_query-feature_proto)**2).sum(2) # torch.Size([525, 35])

    label = torch.arange(num_classes_per_batch).unsqueeze(1).expand(num_classes_per_batch, num_query).flatten()
    label = Variable(label, requires_grad=False).to(device).long()
    
    return nn.CrossEntropyLoss()(logits, label)

def get_feature_nolinear(model, loader_list, device): 
    model.eval()
    feature_all = []
    
    with torch.no_grad(): 
        for i_class, loader in enumerate(loader_list): 
            feature_class = []
            for i_batch, inputs in enumerate(loader): 
                inputs = inputs.to(device, non_blocking=True).float()
                feature = model(inputs)
                feature_class.extend(feature)
                    
                del inputs, feature 
            feature_all.append(torch.stack(feature_class, dim=0))
    feature_all = torch.stack(feature_all)
    return feature_all
