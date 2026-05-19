import copy
import torch
import torch.nn as nn
import numpy as np

from calibration import mmse, KLD_threshold, percentile_threshold
from dataset import load_data
from gcn import GCN, GraphConvolution
from quantization import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#      POST-TRAINING QUANTIZATION
# =========================================================
def ptq(x, qbits, calibration_method="max", signed=True):
    
    if x.is_sparse:
        x = x.to_dense()
        
    x_np = x.detach().cpu().numpy().reshape(-1)
    clip_value = get_clip_value(x_np, qbits, calibration_method, signed)
    
    if signed:
        alpha = -clip_value
        beta = clip_value
        s, z = generate_quantization_qbits_constants(alpha, beta, qbits)
        x_q = quantization_fbits(x, s, z, qbits, ste=False)
    else:
        alpha = 0.0
        beta = clip_value
        s, z = generate_quantization_uqbits_constants(alpha, beta, qbits)
        x_q = quantization_ufbits(x, s, z, qbits, ste=False)
        
    return x_q

# =========================================================
#                Quantized GraphConvolution
# =========================================================
class QuantizedGraphConvolution(nn.Module):
    def __init__(self, layer, num_bits, calibration_method="max", quantize_adj=False):
        super().__init__()
        
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.num_bits = num_bits
        self.calibration_method = calibration_method
        self.quantize_adj = quantize_adj
        
        self.weight = nn.Parameter(layer.weight.detach().clone(), requires_grad=False)
        
        if layer.bias is not None:
            self.bias = nn.Parameter(layer.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
            
        self.quantize_weights()
            
    def quantize_weights(self):
        self.weight.data = ptq(self.weight.data, self.num_bits, self.calibration_method, signed=True)
        
        if self.bias is not None:
            self.bias.data = ptq(self.bias.data, self.num_bits, self.calibration_method, signed=True)
            
    
    def forward(self, x, adj):
        
        x_q = ptq(x, self.num_bits, self.calibration_method, signed=False)
        
        if self.quantize_adj:
            adj_q = ptq(adj, self.num_bits, self.calibration_method, signed=False)
        else:
            adj_q = adj.to_dense() if adj.is_sparse else adj

        support = torch.mm(x_q, self.weight)

        if adj_q.is_sparse:
            output = torch.spmm(adj_q, support)
        else:
            output = torch.mm(adj_q, support)

        if self.bias is not None:
            output = output + self.bias
            
        output = ptq(output, self.num_bits, self.calibration_method, signed=True)
        
        return output
    
# =========================================================
#                    Quantized GCN
# =========================================================
class QuantizedGCN(nn.Module):
    def __init__(self, model, num_bits, calibration_method="max", quantize_adj=False):
        super().__init__()
        self.dropout = model.dropout
        self.layers = nn.ModuleList()
        
        for layer in model.layers:
            if isinstance(layer, GraphConvolution):
                self.layers.append(
                    QuantizedGraphConvolution(
                        layer=layer,
                        num_bits=num_bits,
                        calibration_method=calibration_method,
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

                # ReLU output is non-negative
                x = ptq(
                    x,
                    qbits=layer.num_bits,
                    calibration_method=layer.calibration_method,
                    signed=False
                )

        return torch.log_softmax(x, dim=1)
    
# =========================================================
#                 Quantize whole model
# =========================================================   
def quantize_gcn_model(model, num_bits, calibration_method="max", quantize_adj=False):
    return QuantizedGCN(
        model=copy.deepcopy(model),
        num_bits=num_bits,
        calibration_method=calibration_method,
        quantize_adj=quantize_adj
    )    
    
    
# =========================================================
#                     Evaluation
# =========================================================
@torch.no_grad()
def evaluate_gcn_model(model, adj, features, labels, test_mask, device):
    model.eval()

    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    test_mask = test_mask.to(device)

    outputs = model(features, adj)
    test_outputs = outputs[test_mask]
    test_labels = labels[test_mask]

    pred = test_outputs.argmax(dim=1)
    correct = (pred == test_labels).sum().item()
    total = test_mask.sum().item()

    return 100.0 * correct / total   



# =========================================================
#               Threshold display utility
# =========================================================
def display_quantized_thresholds_gcn(model, features, adj, num_bits):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, GraphConvolution):
            weight_data = layer.weight.detach().cpu().numpy().flatten()

            max_threshold = get_clip_value(weight_data, num_bits, "max", signed=True)
            mse_threshold = get_clip_value(weight_data, num_bits, "mse", signed=True)
            kld_threshold = get_clip_value(weight_data, num_bits, "kld", signed=True)
            pct_threshold = get_clip_value(weight_data, num_bits, "percentile", signed=True)

            print(f"Layer {i} Weights:")
            print(f"  Max Threshold       : {max_threshold:.6f}")
            print(f"  MSE Threshold       : {mse_threshold:.6f}")
            print(f"  KLD Threshold       : {kld_threshold:.6f}")
            print(f"  Percentile Threshold: {pct_threshold:.6f}")

    features_data = features.detach().cpu().numpy().flatten()
    print("\nInput Features:")
    print(f"  Max Threshold       : {get_clip_value(features_data, num_bits, 'max', signed=True):.6f}")
    print(f"  MSE Threshold       : {get_clip_value(features_data, num_bits, 'mse', signed=True):.6f}")
    print(f"  KLD Threshold       : {get_clip_value(features_data, num_bits, 'kld', signed=True):.6f}")
    print(f"  Percentile Threshold: {get_clip_value(features_data, num_bits, 'percentile', signed=True):.6f}")

    adj_dense = adj.to_dense() if adj.is_sparse else adj
    adj_data = adj_dense.detach().cpu().numpy().flatten()
    print("\nAdjacency Matrix:")
    print(f"  Max Threshold       : {get_clip_value(adj_data, num_bits, 'max', signed=False):.6f}")
    print(f"  MSE Threshold       : {get_clip_value(adj_data, num_bits, 'mse', signed=False):.6f}")
    print(f"  KLD Threshold       : {get_clip_value(adj_data, num_bits, 'kld', signed=False):.6f}")
    print(f"  Percentile Threshold: {get_clip_value(adj_data, num_bits, 'percentile', signed=False):.6f}")

           
            
# =========================================================
#                        Main
# =========================================================
if __name__ == "__main__":
    dataset_names = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo']
    num_bits_list = [8, 4, 2, 1]
    calibration_methods = ["max", "mse", "kld", "percentile"]

    for d in dataset_names:
        print("\n" + "=" * 40)
        print(f"Dataset: {d}")
        print("=" * 40)

        adj_norm, features, labels, train_mask, val_mask, test_mask = load_data(d)

        gcn_model = torch.load(f"./models/{d}_gcn.pth", weights_only=False, map_location=device)
        gcn_model.eval()
        gcn_model.to(device)

        float_acc = evaluate_gcn_model(gcn_model, adj_norm, features, labels, test_mask, device)
        print(f"Float model accuracy: {float_acc:.2f}%")

        for n in num_bits_list:
            print(f"\nQuantization with {n}-bit")
            print("-" * 30)

            display_quantized_thresholds_gcn(gcn_model, features, adj_norm, n)

            for method in calibration_methods:
                qmodel = quantize_gcn_model(
                    gcn_model,
                    num_bits=n,
                    calibration_method=method,
                    quantize_adj=False
                ).to(device)

                acc = evaluate_gcn_model(qmodel, adj_norm, features, labels, test_mask, device)
                print(f"{method.upper():>10} PTQ Accuracy: {acc:.2f}%")       
                
                
                
        
        
        
        