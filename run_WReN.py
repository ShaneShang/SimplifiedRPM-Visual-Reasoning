import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset
from torchvision import transforms
import os 
import time
from tqdm import trange

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cuda')

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)

gen_id_list_pre = [
    [1, 16, 20, 34, 37], [8, 12, 24, 36, 39], [5, 17, 21, 33, 38], [3, 10, 29, 31, 37], [0, 14, 27, 35, 38],
    [4, 19, 26, 30, 39], [9, 13, 25, 32, 37], [2, 18, 23, 30, 38], [7, 15, 22, 34, 39], [6, 11, 28, 33, 37]]

##### hyperparameter ##### 
i_split = 1 
gen_id = gen_id_list_pre[i_split]

lr = 0.1
batch_size_train = 1024
start_epoch, total_epochs, print_epoch = 46, 500, 2

from utils_train import * 

save_name = 'WReN_split'+str(i_split)
save_dir, log_file = setup_savename(save_name)

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

loader_train ,loader_val_list = load_data(gen_id, batch_size_train)

from utils_model import panel_SCL
class relation_RN_pos(nn.Module):
    def __init__(self, g_dim=512, f_dim=256, seq_length=3, hidden_dim=80): 
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)) 

        self.proj = nn.Linear(80, int(g_dim/2))
        self.g = nn.Sequential(
            nn.Linear(g_dim, g_dim), nn.ReLU(), 
            nn.Linear(g_dim, f_dim),  nn.ReLU())

        self.f = nn.Sequential(
            nn.Linear(f_dim, f_dim), nn.ReLU(), 
            nn.Linear(f_dim, f_dim), nn.ReLU(), 
            nn.Linear(f_dim, 400))
        self.g_dim = g_dim
        
    def forward(self, x): 
        """x: (batch_size, 3, 80); tags: (batch_size, 3, 3); outputs: (batch_size, 400)"""
        x1 = x + self.pos_embedding
        x2 = self.proj(x1) # torch.Size([13, 3, 256])
        x3 = torch.cat((x2.unsqueeze(1).expand(-1, 3, -1, -1),
                        x2.unsqueeze(2).expand(-1, -1, 3, -1)),dim=3).view(-1, 9, self.g_dim)
        x4 = self.g(x3) # torch.Size([1, 9, 256]), pairwise embedding 
        x5 = torch.sum(x4, dim=1) # torch.Size([1, 256])
        x6 = self.f(x5)
        return x6

class WReN_pos(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        self.module_panel = panel_SCL()
        self.module_relation = relation_RN_pos()
        self.linear = nn.Linear(400, 35)
        
    def forward(self, x):
        """x: (batch_size, 3, 80, 80), outputs: (batch_size, 35); feature: (batch_size, 400)"""
        feature_panel = self.module_panel(x) # torch.Size([13, 3, 512])
        feature_relation = self.module_relation(feature_panel)
        outputs = self.linear(feature_relation)
        return outputs, feature_relation 

model = WReN_pos().to(device)
warmup_epoch = 2 
warmup_lr = np.linspace(1e-8, lr, warmup_epoch*len(loader_train)).reshape((warmup_epoch, -1))
optimizer = torch.optim.SGD( model.parameters(),  lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, min_lr=0.00001, patience=2)

if start_epoch != 0: 
    param_path = save_dir+'last.pt' # load param file
    file = torch.load(param_path)
    model.load_state_dict(file['model_state_dict'])
    optimizer.load_state_dict(file['optimizer'])

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

for current_epoch in range(start_epoch, total_epochs+1):

    feature_test, acc_test, loss_test = get_feature(model, loader_val_list, device)
    torch.save({'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),'feature_test': feature_test}, save_dir+'/'+str(current_epoch)+'.pt')

    if current_epoch < warmup_epoch: 
        acc_train, loss_train, dur = train_epoch(model, loader_train, optimizer, device, warmup_lr[current_epoch])
    else: 
        acc_train, loss_train, dur = train_epoch(model, loader_train, optimizer, device, None)
    
    text = 'E{i_epoch:03d} |acc_train:{acc_train:.1f} |acc_test:{acc_test:.1f} |loss:{loss_train:.8f} |lr:{lr:.6f} |dur:{dur:.2f}\n'.format(
        i_epoch=current_epoch, acc_train=acc_train, acc_test=acc_test, loss_train=loss_train, lr=optimizer.param_groups[0]["lr"], dur=dur)
    with open(log_file, 'a') as f:
        f.write(text)
    print(text)

    torch.save({'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':current_epoch}, save_dir+'/last.pt')

    if current_epoch > warmup_epoch:
        scheduler.step(loss_train)