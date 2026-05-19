import argparse
import numpy as np
from typing import Optional, Union, List
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.datasets import Planetoid, Amazon
import torch_geometric.transforms as T

from brevitas.nn import QuantReLU, QuantIdentity
from brevitas.quant_tensor import QuantTensor
from brevitas.core.bit_width import BitWidthParameter
from brevitas.nn.quant_layer import ActQuantType, BiasQuantType, WeightQuantType
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.inject.enum import ScalingImplType, BitWidthImplType, RestrictValueType
from torch_geometric.utils import scatter as _pyg_scatter
from torch_geometric.utils import softmax as _pyg_softmax
import torch.nn.functional as F
import math


class LearnedIntWeightPerTensorFloat(Int8WeightPerTensorFloat):
    """Per-tensor weight quantizer with learned scale and learned bit-width."""
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    bit_width_impl_type = BitWidthImplType.PARAMETER


class LearnedIntActPerTensorFloat(Int8ActPerTensorFloat):
    """Per-tensor activation quantizer with learned scale (log-space) and learned bit-width."""
    bit_width_impl_type = BitWidthImplType.PARAMETER
    restrict_scaling_type = RestrictValueType.LOG_FP


def scatter_mean(src: Tensor, index: Tensor, dim: int = 0,
                 dim_size: Optional[int] = None) -> Tensor:
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


def scatter_add(src: Tensor, index: Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> Tensor:
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')


def edge_softmax(scores: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
    """Per-destination softmax of edge scores: [E] or [E, H] → same shape."""
    return _pyg_softmax(scores, index=dst, num_nodes=num_nodes, dim=0)


class GCNConvBase(nn.Module):
    """
    Non-quantized GCN convolution base providing weight Parameter storage
    (analogous to nn.Linear for QuantLinear).

    Stores weight [out_features, in_features] and optional bias [out_features].
    The actual forward is handled by the Brevitas quantized subclass.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class QuantGCNConv(QuantWBIOL, GCNConvBase):
    """
    Brevitas-quantized GCN convolution layer.

    This is a proper Brevitas quantized layer (subclasses QuantWBIOL) so all
    weight/input/output quantization is managed by Brevitas internally.

    The graph structure (edge_index) is stored per-forward-call via set_graph()
    or passed directly to forward().

    Args:
        in_features:    Input feature dim (must be divisible by SIMD)
        out_features:   Output feature dim (must be divisible by PE)
        bias:           Use bias (default False for HLS compatibility)
        weight_quant:   Brevitas weight quantizer
        bias_quant:     Brevitas bias quantizer
        input_quant:    Brevitas input quantizer
        output_quant:   Brevitas output quantizer
        return_quant_tensor: Return QuantTensor from forward
        add_self_loops: Add self-loops before aggregation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
        bias_quant: Optional[BiasQuantType] = None,
        input_quant: Optional[ActQuantType] = None,
        output_quant: Optional[ActQuantType] = None,
        add_self_loops: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        GCNConvBase.__init__(self, in_features, out_features, bias,
                             device=device, dtype=dtype)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            # Always False here: inner_forward_impl returns raw Tensor
            # (scatter breaks the IntQuantTensor chain).
            # The model wraps output with QuantReLU for re-quantisation.
            return_quant_tensor=False,
            **kwargs,
        )
        self.add_self_loops = add_self_loops

        # Edge index stored per forward call
        self._edge_index: Optional[Tensor] = None
        self._num_nodes: Optional[int] = None

    # ---- Required abstract properties ----

    @property
    def output_channel_dim(self):
        return 0  # weight shape is [out, in]

    @property
    def channelwise_separable(self) -> bool:
        return False

    @property
    def out_channels(self):
        return self.out_features

    @property
    def per_elem_ops(self):
        return 2 * self.in_features

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):
        max_input_val = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_fc_val = self.weight_quant.max_uint_value(weight_bit_width)
        max_output_val = max_input_val * max_fc_val * self.in_features
        output_bit_width = ceil_ste(torch.log2(max_output_val))
        return output_bit_width

    # ---- Graph setup ----

    def set_graph(self, edge_index: Tensor, num_nodes: Optional[int] = None):
        self._edge_index = edge_index
        self._num_nodes = num_nodes

    # ---- Core computation ----

    def inner_forward_impl(self, x, quant_weight, quant_bias):
        edge_index = self._edge_index
        num_nodes = self._num_nodes

        # Extract raw tensors (scatter doesn't support QuantTensor)
        x_val = x.value if isinstance(x, QuantTensor) else x
        w_val = quant_weight.value if isinstance(quant_weight, QuantTensor) else quant_weight
        b_val = None
        if quant_bias is not None:
            b_val = quant_bias.value if isinstance(quant_bias, QuantTensor) else quant_bias

        if num_nodes is None:
            num_nodes = x_val.shape[0]

        src, dst = edge_index[0], edge_index[1]

        if self.add_self_loops:
            self_loop = torch.arange(num_nodes, device=edge_index.device)
            src = torch.cat([src, self_loop])
            dst = torch.cat([dst, self_loop])

        # Stage 1+2: Aggregate neighbours (mean)
        agg = scatter_mean(x_val[src], dst, dim=0, dim_size=num_nodes)

        # Stage 3: Linear transform (W @ agg + b)
        output = F.linear(agg, w_val, b_val)
        return output

    def forward(self, x: Union[Tensor, QuantTensor],
                edge_index: Optional[Tensor] = None,
                num_nodes: Optional[int] = None) -> Union[Tensor, QuantTensor]:
        """
        Forward with explicit edge_index (more convenient than set_graph).

        Args:
            x:          Node features [N, in_features] or QuantTensor
            edge_index: COO edge list [2, E], or None if set via set_graph()
            num_nodes:  Number of nodes (inferred from x if None)
        """
        if edge_index is not None:
            self.set_graph(edge_index, num_nodes)
        assert self._edge_index is not None, \
            "edge_index must be passed to forward() or set via set_graph()"
        return self.forward_impl(x)


# ── Model ─────────────────────────────────────────────────────────────────────

class LearnedQuantGCN(nn.Module):

    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 init_bit_width: int = 8, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.input_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            bit_width=8,
            return_quant_tensor=True,
        )
        self.conv1 = QuantGCNConv(
            in_features, hidden,
            weight_quant=LearnedIntWeightPerTensorFloat,
            weight_bit_width=init_bit_width,
        )
        self.Qrelu = QuantReLU(
            act_quant=LearnedIntActPerTensorFloat,
            bit_width=init_bit_width,
            return_quant_tensor=True,
        )
        self.relu = nn.ReLU()
        self.conv2 = QuantGCNConv(
            hidden, num_classes,
            weight_quant=LearnedIntWeightPerTensorFloat,
            weight_bit_width=init_bit_width,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_quant(x)
        # x = self.dropout(x.value)
        x = self.conv1(x, edge_index)
        # x = self.Qrelu(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x if not hasattr(x, 'value') else x.value


# ── Bit-width regularization helpers ─────────────────────────────────────────

def collect_bit_widths(model: nn.Module) -> list:
    """Return (name, tensor) for every learned BitWidthParameter in the model."""
    result = []
    for name, module in model.named_modules():
        if isinstance(module, BitWidthParameter):
            result.append((name, module()))
    return result


def bit_width_regularization(model: nn.Module) -> Tensor:
    """
    Mean of all learned bit-widths.

    Minimizing this (via LAMBDA_BW * bit_width_regularization(model)) nudges
    every bit-width downward during QAT, trading accuracy for lower precision.
    Gradient flows through ceil_ste so backprop works through the rounding.
    """
    bws = [bw for _, bw in collect_bit_widths(model)]
    if not bws:
        return torch.tensor(0.0)
    return torch.stack(bws).mean()


def print_bit_widths(model: nn.Module, header: str):
    print(f"\n{header}")
    for name, bw in collect_bit_widths(model):
        print(f"  {name:60s}  {bw.item():.3f} bits")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    PLANETOID = ('Cora', 'Citeseer', 'Pubmed')
    AMAZON    = ('Computers', 'Photo')

    parser = argparse.ArgumentParser(description='Learned-quant GCN on graph datasets')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=PLANETOID + AMAZON,
                        help='Dataset name (default: Cora)')
    parser.add_argument('--epochs',        type=int,   default=300)
    parser.add_argument('--hidden',        type=int,   default=16)
    parser.add_argument('--init_bit_width',type=int,   default=8)
    parser.add_argument('--lambda_bw',     type=float, default=0.003)
    parser.add_argument('--lr',            type=float, default=1e-2)
    parser.add_argument('--seed',          type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    DATASET = args.dataset

    if DATASET in PLANETOID:
        dataset = Planetoid(root='./data', name=DATASET, split='random')
        graph   = dataset[0]
    else:
        dataset = Amazon(root='./data', name=DATASET)
        graph   = T.RandomNodeSplit(num_val=0.1, num_test=0.2)(dataset[0])

    x          = graph.x           # [N, F]
    labels     = graph.y           # [N]
    edge_index = graph.edge_index  # [2, E]
    train_mask = graph.train_mask
    val_mask   = graph.val_mask
    test_mask  = graph.test_mask

    IN_FEATURES = dataset.num_features
    NUM_CLASSES = dataset.num_classes

    dropout      = 0.5 if DATASET in PLANETOID else 0.8
    weight_decay = 5e-4 if DATASET in PLANETOID else 1e-3

    # ── Model, optimizer, loss ────────────────────────────────────────────────
    model     = LearnedQuantGCN(IN_FEATURES, args.hidden, NUM_CLASSES,
                                init_bit_width=args.init_bit_width, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"\nDataset : {DATASET}  ({IN_FEATURES} features, {NUM_CLASSES} classes)")
    print(f"Train   : {train_mask.sum().item()} nodes  |  "
          f"Val: {val_mask.sum().item()}  |  Test: {test_mask.sum().item()}")
    print_bit_widths(model, "=== Learned bit-widths BEFORE training ===")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        logits    = model(x, edge_index)
        task_loss = criterion(logits[train_mask], labels[train_mask])
        bw_loss   = bit_width_regularization(model)
        loss      = task_loss + args.lambda_bw * bw_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_eval = model(x, edge_index)
            train_acc = (logits_eval[train_mask].argmax(1) == labels[train_mask]).float().mean()
            val_acc   = (logits_eval[val_mask].argmax(1)   == labels[val_mask]).float().mean()
            print(f"Epoch {epoch+1:4d} | "
                  f"loss={loss.item():.4f}  "
                  f"task={task_loss.item():.4f}  "
                  f"bw_reg={bw_loss.item():.4f}  "
                  f"train_acc={train_acc:.3f}  "
                  f"val_acc={val_acc:.3f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print_bit_widths(model, "=== Learned bit-widths AFTER training ===")

    model.eval()
    with torch.no_grad():
        logits_eval = model(x, edge_index)
    test_acc = (logits_eval[test_mask].argmax(1) == labels[test_mask]).float().mean()
    print(f"\nTest accuracy: {test_acc:.4f}")

    # ── Inspect learned weight scales ─────────────────────────────────────────
    print("\n=== Learned weight quantization (conv1) ===")
    qt = model.conv1.quant_weight()
    print(f"  scale      : {qt.scale.item():.6f}")
    print(f"  bit_width  : {qt.bit_width.item():.3f}")
    print(f"  signed     : {qt.signed}")

    print("\n=== Learned weight quantization (conv2) ===")
    qt2 = model.conv2.quant_weight()
    print(f"  scale      : {qt2.scale.item():.6f}")
    print(f"  bit_width  : {qt2.bit_width.item():.3f}")
    print(f"  signed     : {qt2.signed}")
