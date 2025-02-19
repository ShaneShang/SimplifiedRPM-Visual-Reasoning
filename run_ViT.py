import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset
from torchvision import transforms
import os 
import time
from tqdm import trange

from utils_ViT import * 

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
start_epoch, total_epochs, print_epoch = 0, 200, 2

##########
save_name = 'ViT_split'+str(i_split)

num_layers = 4
num_heads = 12
hidden_dim = 96
mlp_dim = hidden_dim*4
model = Shane_ViT(num_layers, num_heads, hidden_dim, mlp_dim).to(device)

lr = 0.1
batch_size_train = 1024

gen_id = gen_id_list_pre[i_split]
save_dir, log_file = setup_savename(save_name)
loader_train, loader_val_list = load_data_ViT(gen_id, batch_size_train, device)

warmup_epoch = 2 
warmup_lr = np.linspace(1e-8, lr, warmup_epoch*len(loader_train)).reshape((warmup_epoch, -1))
optimizer = torch.optim.SGD( model.parameters(),  lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, min_lr=0.00001, patience=2)

if start_epoch != 0: 
    param_path = save_dir+'last.pt' # load param file
    file = torch.load(param_path)
    model.load_state_dict(file['model_state_dict'])
    optimizer.load_state_dict(file['optimizer'])

for current_epoch in range(start_epoch, total_epochs+1):

    feature_test, acc_test, loss_test = get_feature(model, loader_val_list, device, True)
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