import numpy as np
import torch

import scipy
from scipy.special import erfc
def H(x):
    return erfc(x/np.sqrt(2))/2

def load_log(log_file): 
    with open(log_file) as f: 
        lines = f.readlines()

    epoch_list = []
    val_acc_list = []
    val_loss_list = []
    gen_SNR_list = []
    
    epoch_all = []
    train_loss_list = []
    lr_list = []
    
    for x in lines: 
        x_list = x.split('|')
        if len(x_list)>1 and x_list[1][0] == 'V': 
            epoch_list.append(float(x_list[0][1:-1]))
            val_acc_list.append(float(x_list[1][8:-1])) # test acc on validation set 
            gen_SNR_list.append(float(x_list[2][8:-1])) # gen SNR 
            val_loss_list.append(float(x_list[3][7:-1])) # test loss 
    
        elif len(x_list)>1 and x_list[1][0] == 'l':
            epoch_all.append(float(x_list[0][1:-1]))
            train_loss_list.append(float(x_list[1][5:-1]))
            lr_list.append(float(x_list[2][3:-1]))
            
    epoch = np.asarray(epoch_list)
    val_acc = np.asarray(val_acc_list)
    val_loss = np.asarray(val_loss_list)
    gen_SNR = np.asarray(gen_SNR_list)
    
    epoch_all = np.asarray(epoch_all)
    train_loss = np.asarray(train_loss_list)
    lr = np.asarray(lr_list)

    return epoch, val_acc, val_loss, gen_SNR, epoch_all, train_loss, lr

def load_x(load_dir, epoch_list): 
    x_list = []
    for i_epoch in epoch_list: 
        x = np.load(load_dir+str(int(i_epoch))+'.npy')
        x_list.append(x)
    return np.asarray(x_list)

def get_error(Xa, Xb, num_trials):
    """Xa, Xb: (num_samples, dim_feature), e.g., [1000, 400] """
    num_samples = Xa.shape[0]
    h_true = []
    for i_trial in range(num_trials): 
        ind_a = torch.randperm(num_samples)
        ind_b = torch.randperm(num_samples)

        Xa, Xb = Xa[ind_a], Xb[ind_b]
        xa, xi_a = Xa[0], Xa[1:]
        xb = Xb[0]

        h = torch.sum((xi_a-xb)**2, axis=1) - torch.sum((xi_a-xa)**2, axis=1)
        h_true.extend(h)
    h_true = torch.stack(h_true).reshape((-1,))
    err_true = torch.mean((h_true<=0).float())
    return err_true

def get_error_list(feature, num_trials=200): 
    """
    feature: (num_classes, num_samples, dim_feature)
    err_list: (num_pairs, num_trials)
    """

    M = len(feature) # num_classes 
    err_list = np.nan * np.zeros((M, M))
    for i in range(M): 
        for j in range(M): 
            if i!=j: 
                err_list[i,j] = get_error(feature[i], feature[j], num_trials)
    return err_list

def get_x_M(x):
    m = x.shape[0]
    x_M = np.nan * np.zeros((m, m))
    for i in range(m):
        c = 0 
        for j in range(m):
            if i!=j:
                x_M[i, j] = x[i, c].item()
                c+=1 
    return x_M

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