
import torch
import torch.optim as optim
import time
import numpy as np
from utils import accuracy
from dataset import load_data
from gcn import GCN
from train import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = 'cpu'
csrData = "./csrData"

params = {
    'dataset': 'citeseer', 
    'times' : 1,
    'seed': 42, 
    'epochs': 200, 
    'lr': 0.01, 
    'weight_decay': 5e-4,
    'hidden': 16, 
    'layer' : 2,
    'dropout': 0.5,
    'save' : False
}


if __name__ == "__main__":

    dataset = ['Cora','Citeseer', 'Pubmed', 'Computers', 'Photo']
    # dataset = ["Cora"]
    

    for data in dataset:
        print('-'*94)
        print(f'{data}')
        print('-'*94)

        
        adj, features, labels, idx_train, idx_val, idx_test = load_data(data)

        if data in ['Computers', 'Photo']:
            params['hidden'] = 16
            params['weight_decay'] = 1e-3
            params['dropout'] = 0.8
    
            

        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)

        
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        
        nclass = labels.max().item() + 1
        acc_lst = list()
        acc_lst2 = list()
        
        print(f"{data} output is : {nclass}")
        
        np.random.seed(params["seed"])
        torch.manual_seed(params["seed"])
        torch.cuda.manual_seed(params["seed"])

        # Model and optimizer
        model = GCN(nfeat=features.shape[1],
                    nhid=params['hidden'],
                    nclass=nclass,
                    dropout=params['dropout'],
                    num_layers= params['layer'])

        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        model.to(device)

        # Train model
        t_total = time.time()

        model = fit(
                model,
                optimizer,
                adj,
                features,
                labels,
                idx_train,
                idx_val,
                epochs=params['epochs'],
                patience=20
            )

        print(f"Total time elapsed: {time.time() - t_total:.4f}s")

        # Testing
        acc_lst.append(test(model, adj, features, labels, idx_test, data))

        if params['save'] == True:
            model_save_path = f"./models/{data}_gcn.pth"
            torch.save(model, model_save_path)

        
            for key, value in model.state_dict().items():
                if key == 'layers.0.weight':
                    save_weights_as_text(model.state_dict(), f"{csrData}/{data}_weights.txt", 'layers.0.weight')
                else:
                    save_weights_as_text(model.state_dict(), f"{csrData}/{data}_weights2.txt", 'layers.1.weight')
