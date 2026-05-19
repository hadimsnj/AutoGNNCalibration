import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class GraphConvolution(Module):
    
    def __init__(self, in_features, out_features, bias = False):
        super(GraphConvolution, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    #-------------------------------------------------------    
    def reset_parameters(self):
        
        stdv = 1. / math.sqrt(self.weight.size(1))
        
        self.weight.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    #-------------------------------------------------------         
    def forward(self, x, adj):
        
        support = torch.mm(x, self.weight)

        if adj.is_sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    #-------------------------------------------------------      
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GCN(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers=2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GraphConvolution(nfeat, nhid))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(nhid, nhid))
        self.layers.append(GraphConvolution(nhid, nclass))
        
        self.dropout = nn.Dropout(p=dropout)
    #-------------------------------------------------------   
    def forward(self, x, adj):

        for i, layer in enumerate(self.layers):
        
            x = layer(x, adj)

            if i != len(self.layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        
        return torch.log_softmax(x, dim=1)


