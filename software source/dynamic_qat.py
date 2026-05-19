import copy
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import torch.nn as nn
import torch.nn.functional as F
from calibration import mmse, KLD_threshold, percentile_threshold
from dataset import load_data
from gcn import GCN, GraphConvolution
from quantization import *
from train import *


# =========================
# STE helpers (LSQ)
# =========================
import math
import torch
import torch.nn as nn


# ===== STE =====
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ===== PACT =====
class PACT(nn.Module):
    def __init__(self, bits, init_alpha=6.0):
        super().__init__()
        self.bits = bits
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x):
        alpha = torch.clamp(self.alpha, min=1e-6)

        x = torch.clamp(x, min=0.0)
        x = torch.minimum(x, alpha)

        qmax = 2 ** self.bits - 1
        scale = alpha / qmax

        x_int = torch.round(x / scale)
        x_int = torch.clamp(x_int, 0, qmax)
        x_q = x_int * scale

        return x + (x_q - x).detach()


# ===== LSQ =====
class LSQWeight(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits

        if bits == 1:
            self.qn = -1
            self.qp = 1
        else:
            self.qn = -2 ** (bits - 1)
            self.qp =  2 ** (bits - 1) - 1

        self.scale = nn.Parameter(torch.tensor(1.0))
        self.initialized = False

    def init_from(self, w):
        with torch.no_grad():
            s = 2 * w.abs().mean() / math.sqrt(self.qp)
            self.scale.copy_(s.clamp(min=1e-6))

    def forward(self, w):
        if not self.initialized:
            self.init_from(w)
            self.initialized = True

        s = torch.clamp(self.scale, min=1e-6)

        g = 1.0 / math.sqrt(w.numel() * self.qp)
        s = GradScale.apply(s, g)

        w_scaled = w / s
        w_clamped = torch.clamp(w_scaled, self.qn, self.qp)
        w_bar = RoundSTE.apply(w_clamped)
        w_q = w_bar * s

        return w + (w_q - w).detach()


# =========================
# GCN Layer (PACT + LSQ)
# =========================
class QATGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bits):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.w_quant = LSQWeight(bits)
        self.act_quant = PACT(bits)

    def forward(self, x, adj, apply_act=True):
        w_q = self.w_quant(self.weight)

        x = torch.mm(x, w_q)
        x = torch.mm(adj, x)  # dense adj for simplicity
        x = x + self.bias

        if apply_act:
            x = F.relu(x)
            x = self.act_quant(x)

        return x


# =========================
# Full Model
# =========================
class QATGraphConv(nn.Module):
    def __init__(self, layer, bits):
        super().__init__()
    
        self.weight = nn.Parameter(layer.weight.detach().clone())
        if layer.bias is not None:
            self.bias = nn.Parameter(layer.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.w_quant = LSQWeight(bits)
        
        # self.x_quant = PACT(bits)
        # self.adj_quant = PACT(bits)

        self.act_quant = PACT(bits)

    def forward(self, x, adj, apply_act=True):

        # x = self.x_quant(x)

        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj

        # adj_dense = self.adj_quant(adj_dense)

        w_q = self.w_quant(self.weight)

        # matmul
        x = torch.mm(x, w_q)
        x = torch.mm(adj_dense, x)  

        if self.bias is not None:
            x = x + self.bias

        if apply_act:
            x = F.relu(x)
            x = self.act_quant(x)

        return x


class PACTLSQGCN(nn.Module):
    def __init__(self, float_model, bits):
        super().__init__()

        self.layers = nn.ModuleList([
            QATGraphConv(layer, bits)
            for layer in float_model.layers
        ])

        self.dropout = float_model.dropout

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj, apply_act=False)

            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
                x = layer.act_quant(x)

        return torch.log_softmax(x, dim=1)


# =========================
# MAIN (example usage)
# =========================
if __name__ == "__main__":
    dataset_names = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo']
    num_bits_list = [8, 4, 2]
    
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
        'save' : False,
        'patience': 20,
    }

    for d in dataset_names:
        print("\n" + "=" * 40)
        print(f"Dataset: {d}")
        print("=" * 40)

        adj_norm, features, labels, train_mask, val_mask, test_mask = load_data(d)
        
        if d in ['Computers', 'Photo']:
            params['hidden'] = 16
            params['weight_decay'] = 1e-3
            params['dropout'] = 0.8

        if d in ['Pubmed', 'Computers', 'Photo']:
            params['patience'] = 100


        features = features.to(device)
        adj_norm = adj_norm.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

        np.random.seed(params["seed"])
        torch.manual_seed(params["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params["seed"])

        nclass = labels.max().item() + 1
        # print(f"{d} output is : {nclass}")

        # -------------------------------------------------
        # Train float model first
        # -------------------------------------------------
        float_model = GCN(
            nfeat=features.shape[1],
            nhid=params['hidden'],
            nclass=nclass,
            dropout=params['dropout'],
            num_layers=params['layer']
        ).to(device)

        optimizer = optim.Adam(
            float_model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        print("\n[Float Training]")
        t_total = time.time()


     
        float_model, history = fit(
            float_model,
            optimizer,
            adj_norm,
            features,
            labels,
            train_mask,
            val_mask,
            epochs=params['epochs'],
            patience=params['patience'],
            monitor="val_acc"
        )
        print(f"Total float training time: {time.time() - t_total:.4f}s")

        float_acc = test(float_model, adj_norm, features, labels, test_mask, d)
        print(f"Float model accuracy: {float_acc:.4f}")


        lr = params['lr']
        wd = params['weight_decay']
        # ===== QAT with PACT + LSQ =====
        for nbits in num_bits_list:
            print(f"\n--- {nbits}-bit PACT+LSQ ---")

            qat_model = PACTLSQGCN(float_model, bits=nbits).to(features.device)

            # separate LR (important)
            quant_params, main_params = [], []
            for name, p in qat_model.named_parameters():
                if "scale" in name or "alpha" in name:
                    quant_params.append(p)
                else:
                    main_params.append(p)

            qat_optimizer = optim.Adam([
                {"params": main_params, "lr": lr, "weight_decay": wd},
                {"params": quant_params, "lr": lr * 0.1, "weight_decay": 0.0},
            ])
                
            qat_model, history = fit(
                qat_model,
                qat_optimizer,
                adj_norm,
                features,
                labels,
                train_mask,
                val_mask,
                epochs=params['epochs'],
                patience=params['patience'],
                monitor="val_acc"
            )

            acc = test(qat_model, adj_norm, features, labels, test_mask,
                    f"{d}_pact_lsq_{nbits}bit")

            print(f"{nbits}-bit Acc: {acc:.4f}")