import numpy as np
import scipy.sparse as sp
import torch



def normalize_adj(adj):
    
    adj = sp.coo_matrix(adj)
    
    rowsum = np.array(adj.sum(1)) #D
    
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() #D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0 
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) #D^-0.5
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() #D^-0.5AD^0.5



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)


def accuracy(pred, targ):
    pred = torch.max(pred, 1)[1]
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    
    return ac


def save_as_text(my_list, name, s, csrData:str = './csrData'):
    tensor_cpu = my_list.cpu()
    tensor_list = tensor_cpu.tolist()
    with open(f'{csrData}/{name}.txt', 'w') as file:
        for item in tensor_list:
            if name == f"{s}_output":
                for i in item:
                    file.write(f"{i}\n")
            else:
                file.write(f"{item}\n") 


def save_weights_as_text(state_dict, filename, name):
    with open(filename, 'w') as f:
        for key, value in state_dict.items():
            if key == name:
                np.savetxt(f, value.cpu().numpy(), delimiter=', ', newline='\n', fmt='%.15f')
                f.write("\n")