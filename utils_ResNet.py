import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset
from torchvision import transforms
from utils_train import * 
import time

from torch.utils.data import Dataset, DataLoader
def load_data(gen_id, batch_size_train):
    attr_train_all = torch.load('data/attr_train.pt') # torch.Size([40, 10000, 3, 9, 3])
    attr_val_all = torch.load('data/attr_val.pt')
    attr_test_all = torch.load('data/attr_test.pt')
    
    train_id = [i for i in range(40) if i not in gen_id]
    train_inputs = attr_train_all[train_id] # torch.Size([35, 10000, 3, 9, 3])
    val_inputs = attr_val_all[train_id]

    dataset_train = dataset_PGM(train_inputs)

    loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True, pin_memory=True)

    dataset_val_list = [dataset_PGM_single(x) for x in val_inputs]
    loader_val_list = [DataLoader(d, batch_size=500, shuffle=False, pin_memory=True) for d in dataset_val_list]

    return loader_train, loader_val_list

from torchvision.models import resnet50
class ModifiedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*list(resnet50().children())[:-1])  # Exclude the original FC layer
        self.fc = nn.Linear(2048, 35)  # Replace with your desired output classes if necessary

    def forward(self, x):
        x = self.features(x)
        feature = torch.flatten(x, 1)  # Flatten the output for the FC layer
        outputs = self.fc(feature)
        return outputs, feature

def get_feature(model, loader_list, device): 
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
 
    acc = 100.0 * correct / total 
    loss_CE = running_loss_CE / len(loader_list)

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
