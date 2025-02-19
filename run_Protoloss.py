import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 

import os 
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cuda')

from utils_train import * 
from utils_train_nolinear import * 
from utils_model import SCL
from functools import partial

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)

gen_id_list_pre = [
    [1, 16, 20, 34, 37], [8, 12, 24, 36, 39], [5, 17, 21, 33, 38], [3, 10, 29, 31, 37], [0, 14, 27, 35, 38],
    [4, 19, 26, 30, 39], [9, 13, 25, 32, 37], [2, 18, 23, 30, 38], [7, 15, 22, 34, 39], [6, 11, 28, 33, 37]]

############### hyperparameter ############### 
i_split = 8
gen_id = gen_id_list_pre[i_split]
save_name = 'Proto_split'+str(i_split)
start_epoch, total_epochs, print_epoch = 0, 50, 5

num_classes_per_batch = 32 
num_samples_per_class = 16
num_batches = 684 

sampler_train = Sampler_class(num_batches, num_classes_per_batch, num_samples_per_class, 35, 10000)
############### run ############### 
loss_fn = partial(loss_fn_Proto, device=device, num_classes_per_batch=num_classes_per_batch, 
                  num_samples_per_class=num_samples_per_class)
save_dir, log_file = setup_savename(save_name)
loader_train, loader_val_list, loader_gen_list, sampler_train = load_data(gen_id, sampler_train=sampler_train)

model = SCL().to(device)
lr = 0.01

warmup_epoch = 2 
warmup_lr = np.linspace(1e-8, lr, warmup_epoch*len(loader_train)).reshape((warmup_epoch, -1))

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, min_lr=0.00001, patience=2)

if start_epoch != 0: 
    param_path = save_dir+'last.pt' # load param file
    file = torch.load(param_path)
    model.load_state_dict(file['model_state_dict'])
    optimizer.load_state_dict(file['optimizer'])

############### run ############### 
for current_epoch in range(start_epoch, total_epochs+1):

    sampler_train.set_epoch(current_epoch) 
    
    if current_epoch % print_epoch == 0:

        feature_gen = get_feature_nolinear(model, loader_gen_list, device)
        torch.save({'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'feature_gen': feature_gen}, save_dir+'/'+str(current_epoch)+'.pt')

    start_time = time.time()
    if current_epoch < warmup_epoch: 
        loss_train = train_epoch_nonlinear(model, loss_fn, loader_train, optimizer, device, warmup_lr[current_epoch])
    else: 
        loss_train = train_epoch_nonlinear(model, loss_fn, loader_train, optimizer, device, None)
    dur = time.time() - start_time

    text = 'E{i_epoch:03d} |loss:{loss_train:.8f} |lr:{lr:.6f} |dur:{dur:.2f}\n'.format(
        i_epoch=current_epoch, loss_train=loss_train, lr=optimizer.param_groups[0]["lr"], dur=dur)
    with open(log_file, 'a') as f:
        f.write(text)
    print(text) 

    torch.save({'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':current_epoch}, 
               save_dir+'/last.pt')
    
    if current_epoch > warmup_epoch:
        scheduler.step(loss_train)