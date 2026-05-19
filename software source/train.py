import copy
import torch
import torch.optim as optim
import time
import numpy as np
from utils import *
from dataset import load_data
from gcn import GCN


criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = 'cpu'




def train(epoch, model, optimizer, adj, features, labels, idx_train):
    t = time.time()
    
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    
    loss = criterion(output[idx_train], labels[idx_train])
    acc = accuracy(output[idx_train], labels[idx_train])
    
    loss.backward()
    
    optimizer.step()

    
    return loss.item(), acc

@torch.no_grad()
def evaluate(model, adj, features, labels, idx):
    model.eval()

    output = model(features, adj)

    loss = criterion(output[idx], labels[idx])
    acc = accuracy(output[idx], labels[idx])

    return loss.item(), acc


def fit(
    model,
    optimizer,
    adj,
    features,
    labels,
    idx_train,
    idx_val,
    epochs=200,
    patience=20,
    monitor="val_loss",   # "val_loss" or "val_acc"
    min_delta=0.0,
    verbose=False,
):
    best_state = None
    patience_counter = 0

    if monitor == "val_loss":
        best_metric = float("inf")
    elif monitor == "val_acc":
        best_metric = float("-inf")
    else:
        raise ValueError("monitor must be 'val_loss' or 'val_acc'")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train(
            epoch, model, optimizer, adj, features, labels, idx_train
        )
        val_loss, val_acc = evaluate(
            model, adj, features, labels, idx_val
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

        improved = False
        if monitor == "val_loss":
            if val_loss < (best_metric - min_delta):
                best_metric = val_loss
                improved = True
        else:  # monitor == "val_acc"
            if val_acc > (best_metric + min_delta):
                best_metric = val_acc
                improved = True

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch:03d}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history



@torch.no_grad()
def test(model, adj, features, labels, idx_test, name):
    
    model.eval()
    
    output = model(features, adj)
    
    # save_as_text(output, f'{name}_output', name)
    # save_as_text(labels, f'{name}_labels', name)
    
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # print(f"Test set results:",
    #       f"loss= {loss_test.item():.4f}",
    #       f"accuracy= {acc_test:.4f}")
    
    return acc_test