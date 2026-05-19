from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, TUDataset
import torch_geometric.transforms as T
import torch_sparse
import scipy.sparse as sp
from utils import normalize_adj, sparse_mx_to_torch_sparse_tensor


def load_data(name:str = "Cora", root:str="./data", csrData:str = "./csrData"):

    assert name in ['Cora','Citeseer', 'Pubmed', 'Computers', 'Photo']
    
    if name in ['Cora','Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=name, split='random')
    
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root=root, name=name)

    graph = dataset[0]
    print(graph)
    
    features = graph.x
    adj = graph.edge_index
    labels = graph.y

    if name in ['Cora','Citeseer', 'Pubmed']:
        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask

    else:
        split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
        graph = split(graph)

        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask


    
    adj = torch_sparse.SparseTensor(row = adj[0], col = adj[1])
    
    adj = adj.to_scipy(layout='coo')
    
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))    
    
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    
    
    #save feature in csr foramt
    fea_csr = sp.csr_matrix(features.to_dense().numpy())
    data, indices, indptr = fea_csr.data, fea_csr.indices, fea_csr.indptr
    # with open(f'{csrData}/{name}_fea.txt', 'w') as f:
    #     f.write(', '.join(map(str, indptr)))
    #     f.write('\n')
    #     f.write(', '.join(map(str, indices)))
    #     f.write('\n')
    #     f.write(', '.join(map(str, data)))
    #     f.write('\n')
    
    #save adjacency in csr foramt
    adj_norm_csr = sp.csr_matrix(adj_norm.to_dense().numpy())
    data, indices, indptr = adj_norm_csr.data, adj_norm_csr.indices, adj_norm_csr.indptr
    # with open(f'{csrData}/{name}_adj.txt', 'w') as f:
    #     f.write(', '.join(map(str, indptr)))
    #     f.write('\n')
    #     f.write(', '.join(map(str, indices)))
    #     f.write('\n')
    #     f.write(', '.join(map(str, data)))
    #     f.write('\n')
    

  
    return adj_norm, features, labels, train_mask, val_mask, test_mask


# for name in ['Cora','Citeseer', 'Pubmed', 'Computers', 'Photo']:
#     load_data(name)

