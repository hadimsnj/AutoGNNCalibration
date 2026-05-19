import copy
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time

from calibration import mmse, KLD_threshold, percentile_threshold
from dataset import load_data
from gcn import GCN, GraphConvolution
from quantization import *
from train import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# device = 'cpu'
csrData = "./csrData"

# =========================================================
#                EMA Observer
# =========================================================
class EMAObserver(nn.Module):
    def __init__(self, momentum=0.95, signed=True, eps=1e-8):
        super().__init__()
        self.momentum = momentum
        self.signed = signed
        self.eps = eps

        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.enabled = True

    @torch.no_grad()
    def update(self, x):
        if not self.enabled:
            return

        if x.is_sparse:
            x = x.to_dense()

        if self.signed:
            cur_min = x.min()
            cur_max = x.max()
        else:
            cur_min = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            cur_max = x.max()

        if torch.isinf(self.min_val) or torch.isinf(self.max_val):
            self.min_val.copy_(cur_min)
            self.max_val.copy_(cur_max)
        else:
            self.min_val.mul_(self.momentum).add_(cur_min * (1 - self.momentum))
            self.max_val.mul_(self.momentum).add_(cur_max * (1 - self.momentum))

    def get_range(self):
        min_val = self.min_val
        max_val = self.max_val

        if self.signed:
            clip = torch.max(min_val.abs(), max_val.abs())
            clip = torch.clamp(clip, min=self.eps)
            alpha = -clip
            beta = clip
        else:
            alpha = torch.tensor(0.0, device=max_val.device, dtype=max_val.dtype)
            beta = torch.clamp(max_val, min=self.eps)

        return alpha, beta

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

        

# =========================================================
#                Calibration helper
# =========================================================
def get_clip_value(values, num_bits, calibration_method="max", signed=True):
    values = np.array(values, dtype=np.float32)
    
    if values.size == 0:
        return 1e-6
    
    if calibration_method == "max":
        clip_value = np.max(np.abs(values)) if signed else np.max(values)
    
    elif calibration_method == "mse":
        clip_value = mmse(values, num_bits, signed=signed)

    elif calibration_method == "kld":
        clip_value = KLD_threshold(values, num_bits)

    elif calibration_method == "percentile":
        clip_value = percentile_threshold(values, percentile=99.9, symmetric=signed)

    else:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")
    
    if clip_value == 0 or not np.isfinite(clip_value):
        clip_value = 1e-6
        
    return float(clip_value)

# =========================================================
#                Static range helper
# =========================================================
def static_range(x, num_bits, calibration_method="max", signed=True):
    if x.is_sparse:
        x = x.to_dense()

    x_np = x.detach().cpu().numpy().reshape(-1)
    clip_value = get_clip_value(x_np, num_bits, calibration_method, signed)

    if signed:
        alpha = -clip_value
        beta = clip_value
    else:
        alpha = 0.0
        beta = clip_value

    return alpha, beta


# =========================================================
#                Fake quantizer for QAT
# =========================================================
def fake_quantize(x, alpha, beta, qbits, signed=True):
    if x.is_sparse:
        x = x.to_dense()

    if signed:
        s, z = generate_quantization_qbits_constants(alpha, beta, qbits)
        return quantization_fbits(x, s, z, qbits, ste=True)
    else:
        s, z = generate_quantization_uqbits_constants(alpha, beta, qbits)
        return quantization_ufbits(x, s, z, qbits, ste=True)
    


# =========================================================
#         Dynamic Quantized GraphConvolution 
# =========================================================
class DynamicGraphConvolution(nn.Module):
    def __init__(self, layer, num_bits, momentum=0.95, quantize_adj=False):
        super().__init__()
        
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.num_bits = num_bits
        self.quantize_adj = quantize_adj
        
        self.weight = nn.Parameter(layer.weight.detach().clone())
        
        if layer.bias is not None:
            self.bias = nn.Parameter(layer.bias.detach().clone())
        else:
            self.register_parameter("bias", None)
            
        self.w_observer = EMAObserver(momentum=momentum, signed=True)
        self.x_observer = EMAObserver(momentum=momentum, signed=False)
        self.out_observer = EMAObserver(momentum=momentum, signed=True)
        
        if quantize_adj:
            self.adj_observer = EMAObserver(momentum=momentum, signed=False)
        else:
            self.adj_observer = None
            
    def forward(self, x, adj):
        if self.training:
            self.x_observer.update(x)
            self.w_observer.update(self.weight)        
            
            if self.quantize_adj:
                self.adj_observer.update(adj)
        
        x_alpha, x_beta = self.x_observer.get_range()
        w_alpha, w_beta = self.w_observer.get_range()

        x_q = fake_quantize(x, x_alpha.item(), x_beta.item(), self.num_bits, signed=False)
        w_q = fake_quantize(self.weight, w_alpha.item(), w_beta.item(), self.num_bits, signed=True)
        
        if self.quantize_adj:
            a_alpha, a_beta = self.adj_observer.get_range()
            adj_q = fake_quantize(adj, a_alpha.item(), a_beta.item(), self.num_bits, signed=False)
        else:
            adj_q = adj.to_dense() if adj.is_sparse else adj
            
        support = torch.mm(x_q, w_q)
        
        if adj_q.is_sparse:
            output = torch.spmm(adj_q, support)
        else:
            output = torch.mm(adj_q, support)
            
        if self.bias is not None:
            if self.training:
                self.out_observer.update(self.bias)
            output = output + self.bias

        if self.training:
            self.out_observer.update(output)
                
        out_alpha, out_beta = self.out_observer.get_range()
        output = fake_quantize(output, out_alpha.item(), out_beta.item(), self.num_bits, signed=True)
    
        return output
    
    
# =========================================================
#         Static Quantized GraphConvolution 
# =========================================================
class StaticGraphConvolution(nn.Module):
    def __init__(self, layer, num_bits, calibration_method="max", quantize_adj=False):
        super().__init__()

        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.num_bits = num_bits
        self.calibration_method = calibration_method
        self.quantize_adj = quantize_adj

        self.weight = nn.Parameter(layer.weight.detach().clone())

        if layer.bias is not None:
            self.bias = nn.Parameter(layer.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        # fixed per-layer weight range
        w_alpha, w_beta = static_range(self.weight.data, num_bits, calibration_method, signed=True)
        self.register_buffer("w_alpha", torch.tensor(w_alpha))
        self.register_buffer("w_beta", torch.tensor(w_beta))

        if self.bias is not None:
            b_alpha, b_beta = static_range(self.bias.data, num_bits, calibration_method, signed=True)
            self.register_buffer("b_alpha", torch.tensor(b_alpha))
            self.register_buffer("b_beta", torch.tensor(b_beta))

        # fixed per-layer activation ranges (initialized once)
        self.register_buffer("x_alpha", torch.tensor(0.0))
        self.register_buffer("x_beta", torch.tensor(0.0))
        self.register_buffer("out_alpha", torch.tensor(0.0))
        self.register_buffer("out_beta", torch.tensor(0.0))
        self.ranges_initialized = False

        if self.quantize_adj:
            self.register_buffer("a_alpha", torch.tensor(0.0))
            self.register_buffer("a_beta", torch.tensor(0.0))

    def _init_ranges(self, x, adj):
        x_alpha, x_beta = static_range(x, self.num_bits, self.calibration_method, signed=False)
        self.x_alpha.fill_(x_alpha)
        self.x_beta.fill_(x_beta)

        if self.quantize_adj:
            a_alpha, a_beta = static_range(adj, self.num_bits, self.calibration_method, signed=False)
            self.a_alpha.fill_(a_alpha)
            self.a_beta.fill_(a_beta)

        # estimate output range once using quantized input/weight
        x_q = fake_quantize(x, self.x_alpha.item(), self.x_beta.item(), self.num_bits, signed=False)
        w_q = fake_quantize(self.weight, self.w_alpha.item(), self.w_beta.item(), self.num_bits, signed=True)

        if self.quantize_adj:
            adj_q = fake_quantize(adj, self.a_alpha.item(), self.a_beta.item(), self.num_bits, signed=False)
        else:
            adj_q = adj.to_dense() if adj.is_sparse else adj

        support = torch.mm(x_q, w_q)
        output = torch.spmm(adj_q, support) if adj_q.is_sparse else torch.mm(adj_q, support)

        if self.bias is not None:
            b_q = fake_quantize(self.bias, self.b_alpha.item(), self.b_beta.item(), self.num_bits, signed=True)
            output = output + b_q

        out_alpha, out_beta = static_range(output, self.num_bits, self.calibration_method, signed=True)
        self.out_alpha.fill_(out_alpha)
        self.out_beta.fill_(out_beta)

        self.ranges_initialized = True

    def forward(self, x, adj):
        if not self.ranges_initialized:
            self._init_ranges(x, adj)

        x_q = fake_quantize(x, self.x_alpha.item(), self.x_beta.item(), self.num_bits, signed=False)
        w_q = fake_quantize(self.weight, self.w_alpha.item(), self.w_beta.item(), self.num_bits, signed=True)

        if self.quantize_adj:
            adj_q = fake_quantize(adj, self.a_alpha.item(), self.a_beta.item(), self.num_bits, signed=False)
        else:
            adj_q = adj.to_dense() if adj.is_sparse else adj

        support = torch.mm(x_q, w_q)
        output = torch.spmm(adj_q, support) if adj_q.is_sparse else torch.mm(adj_q, support)

        if self.bias is not None:
            b_q = fake_quantize(self.bias, self.b_alpha.item(), self.b_beta.item(), self.num_bits, signed=True)
            output = output + b_q

        output = fake_quantize(output, self.out_alpha.item(), self.out_beta.item(), self.num_bits, signed=True)
        return output
    
# =========================================================
#                    Dynamic GCN
# =========================================================
class DynamicGCN(nn.Module):
    def __init__(self, model, num_bits, momentum=0.95, quantize_adj=False):
        super().__init__()
        self.dropout = model.dropout
        self.layers = nn.ModuleList()

        for layer in model.layers:
            if isinstance(layer, GraphConvolution):
                self.layers.append(
                    DynamicGraphConvolution(
                        layer=layer,
                        num_bits=num_bits,
                        momentum=momentum,
                        quantize_adj=quantize_adj
                    )
                )
            else:
                raise TypeError("Only GraphConvolution layers are supported.")

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)

            if i != len(self.layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)

        return torch.log_softmax(x, dim=1)
    
# =========================================================
#                    Static QAT GCN
# =========================================================
class StaticGCN(nn.Module):
    def __init__(self, model, num_bits, calibration_method="max", quantize_adj=False):
        super().__init__()
        self.dropout = model.dropout
        self.layers = nn.ModuleList()

        for layer in model.layers:
            if isinstance(layer, GraphConvolution):
                self.layers.append(
                    StaticGraphConvolution(
                        layer=layer,
                        num_bits=num_bits,
                        calibration_method=calibration_method,
                        quantize_adj=quantize_adj
                    )
                )
            else:
                raise TypeError("Only GraphConvolution layers are supported.")

        self.num_bits = num_bits
        self.calibration_method = calibration_method

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)

            if i != len(self.layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
                
                alpha, beta = static_range(
                    x, self.num_bits, self.calibration_method, signed=False
                )
                x = fake_quantize(x, alpha, beta, self.num_bits, signed=False)
        
        return torch.log_softmax(x, dim=1)

    
    
# =========================================================
#                    Builders
# =========================================================
def build_dynamic_model(model, num_bits, momentum=0.95, quantize_adj=False):
    return DynamicGCN(
        model=copy.deepcopy(model),
        num_bits=num_bits,
        momentum=momentum,
        quantize_adj=quantize_adj
    )


def build_static_model(model, num_bits, calibration_method="max", quantize_adj=False):
    return StaticGCN(
        model=copy.deepcopy(model),
        num_bits=num_bits,
        calibration_method=calibration_method,
        quantize_adj=quantize_adj
    )
    
    
# =========================================================
#                        Main 
# =========================================================
if __name__ == "__main__":
    dataset_names = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo']
    # dataset_names = ['Cora']
    num_bits_list = [8, 4, 2, 1]
    # calibration_methods = ["max", "mse", "kld", "percentile"]
    calibration_methods = ["max", "mse", "kld", "percentile"]
    print(device)

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
        gcn_model = GCN(
            nfeat=features.shape[1],
            nhid=params['hidden'],
            nclass=nclass,
            dropout=params['dropout'],
            num_layers=params['layer']
        ).to(device)

        optimizer = optim.Adam(
            gcn_model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        print("\n[Float Training]")
        t_total = time.time()

        float_model, history = fit(
            gcn_model,
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

        float_acc = test(gcn_model, adj_norm, features, labels, test_mask, d)
        print(f"Float model accuracy: {float_acc:.4f}")

        # -------------------------------------------------
        # QAT 
        # -------------------------------------------------
        for n in num_bits_list:
            print(f"\nQuantization with {n}-bit")
            print("-" * 30)

            # ------------------------------
            # Dynamic QAT (EMA)
            # ------------------------------
            print("\n[Dynamic QAT - EMA]")

            qat_model = build_dynamic_model(
                gcn_model,
                num_bits=n,
                momentum=0.95,
                quantize_adj=True
            ).to(device)

            qat_optimizer = optim.Adam(
                qat_model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )

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

            for m in qat_model.modules():
                if isinstance(m, EMAObserver):
                    m.disable()

            acc = test(qat_model, adj_norm, features, labels, test_mask, f"{d}_qat_dynamic_{n}bit")
            print(f"{'EMA'} QAT Accuracy: {acc:.4f}")

            # ------------------------------
            # Static QAT
            # ------------------------------
            print("\n[Static QAT]")

            for method in calibration_methods:
                qat_model = build_static_model(
                    gcn_model,
                    num_bits=n,
                    calibration_method=method,
                    quantize_adj=True
                ).to(device)

                qat_optimizer = optim.Adam(
                    qat_model.parameters(),
                    lr=params['lr'],
                    weight_decay=params['weight_decay']
                )

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

                acc = test(qat_model, adj_norm, features, labels, test_mask, f"{d}_qat_{method}_{n}bit")
                print(f"{method.upper()} QAT Accuracy: {acc:.4f}")