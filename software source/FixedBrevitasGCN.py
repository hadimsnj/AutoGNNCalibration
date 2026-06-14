import argparse
import copy
import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid, Amazon
import torch_geometric.transforms as T

from brevitas.nn import QuantIdentity
from brevitas.quant_tensor import QuantTensor
from brevitas.nn.quant_layer import ActQuantType, BiasQuantType, WeightQuantType
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.quant.binary import SignedBinaryWeightPerTensorConst, SignedBinaryActPerTensorConst
from brevitas.inject.enum import ScalingImplType
from torch_geometric.utils import scatter as _pyg_scatter


class BinaryWeightPerTensorFloat(SignedBinaryWeightPerTensorConst):
    """Binary weights {-1, +1} * scale, scale learned as a parameter."""
    scaling_impl_type = ScalingImplType.PARAMETER


class BinaryActPerTensorFloat(SignedBinaryActPerTensorConst):
    """Binary activations {-1, +1} * scale, scale learned as a parameter."""
    scaling_impl_type = ScalingImplType.PARAMETER


# ── Scatter helper ────────────────────────────────────────────────────────────

def scatter_mean(src: Tensor, index: Tensor, dim: int = 0,
                 dim_size: Optional[int] = None) -> Tensor:
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


# ── Weight-parameter base (mirrors nn.Linear storage for QuantWBIOL) ─────────

class GCNConvBase(nn.Module):
    """Holds weight [out, in] and optional bias [out] as nn.Parameters."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, device=None, dtype=None):
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
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


# ── Brevitas-quantized GCN convolution ───────────────────────────────────────

class QuantGCNConv(QuantWBIOL, GCNConvBase):
    """
    Fixed-precision Brevitas GCN convolution.

    Quantizes weights (and optionally inputs/outputs) with a constant bit width
    chosen at construction time.  The graph topology is injected via
    ``edge_index`` each forward pass.

    The aggregation stage uses mean-pooling over neighbours:
        agg_i = mean_{j in N(i)} x_j
    followed by a linear transform:
        out_i = W @ agg_i  (+  bias)
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
            return_quant_tensor=False,
            **kwargs,
        )
        self.add_self_loops = add_self_loops
        self._edge_index: Optional[Tensor] = None
        self._num_nodes: Optional[int] = None

    # ── Required abstract properties ─────────────────────────────────────────

    @property
    def output_channel_dim(self) -> int:
        return 0  # weight shape is [out, in]

    @property
    def channelwise_separable(self) -> bool:
        return False

    @property
    def out_channels(self) -> int:
        return self.out_features

    @property
    def per_elem_ops(self) -> int:
        return 2 * self.in_features

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):
        return int(input_bit_width + weight_bit_width + math.ceil(math.log2(self.in_features + 1)))

    # ── Graph injection ───────────────────────────────────────────────────────

    def set_graph(self, edge_index: Tensor, num_nodes: Optional[int] = None):
        self._edge_index = edge_index
        self._num_nodes = num_nodes

    # ── Core computation (called by Brevitas's forward_impl) ─────────────────

    def inner_forward_impl(self, x, quant_weight, quant_bias):
        edge_index = self._edge_index
        num_nodes = self._num_nodes

        # Unwrap QuantTensor if needed (scatter doesn't support it)
        x_val = x.value if isinstance(x, QuantTensor) else x
        w_val = quant_weight.value if isinstance(quant_weight, QuantTensor) else quant_weight
        b_val = (quant_bias.value if isinstance(quant_bias, QuantTensor) else quant_bias) \
                if quant_bias is not None else None

        if num_nodes is None:
            num_nodes = x_val.shape[0]

        src, dst = edge_index[0], edge_index[1]

        if self.add_self_loops:
            self_loop = torch.arange(num_nodes, device=edge_index.device)
            src = torch.cat([src, self_loop])
            dst = torch.cat([dst, self_loop])

        # Aggregate (mean over in-neighbours) then linear transform
        agg = scatter_mean(x_val[src], dst, dim=0, dim_size=num_nodes)
        return F.linear(agg, w_val, b_val)

    def forward(self, x: Union[Tensor, QuantTensor],
                edge_index: Optional[Tensor] = None,
                num_nodes: Optional[int] = None) -> Tensor:
        if edge_index is not None:
            self.set_graph(edge_index, num_nodes)
        assert self._edge_index is not None, \
            "Provide edge_index to forward() or call set_graph() first."
        return self.forward_impl(x)


# ── Fixed-precision 2-layer GCN ──────────────────────────────────────────────

class FixedQuantGCN(nn.Module):
    """
    Two-layer GCN where both weights and input activations are quantized
    to a fixed ``bit_width`` using Brevitas.

    Architecture:
        QuantIdentity  →  QuantGCNConv(bw)  →  ReLU  →  Dropout
                       →  QuantGCNConv(bw)  →  logits
    """

    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 bit_width: int = 8, dropout: float = 0.5):
        super().__init__()
        assert bit_width in (1, 2, 4, 8), f"Unsupported bit_width={bit_width}; choose from 1,2,4,8"
        self.bit_width = bit_width
        self.dropout = nn.Dropout(p=dropout)

        # 1-bit: use true binary quantizers {-scale, +scale}.
        # 2/4/8-bit: use Int8 (signed integer) quantizers with overridden bit_width.
        # NOTE: Int8WeightPerTensorFloat with weight_bit_width=1 is degenerate
        # (NarrowIntQuant signed 1-bit → range [0,0] → scale=inf → NaN).
        if bit_width == 1:
            act_quant   = BinaryActPerTensorFloat
            weight_quant = BinaryWeightPerTensorFloat
            act_kwargs  = {}          # binary quant sets bit_width internally
            weight_kwargs = {}        # same
        else:
            act_quant   = Int8ActPerTensorFloat
            weight_quant = Int8WeightPerTensorFloat
            act_kwargs  = {'bit_width': bit_width}
            weight_kwargs = {'weight_bit_width': bit_width}

        # Quantize the raw input features to the target precision
        self.input_quant = QuantIdentity(
            act_quant=act_quant,
            return_quant_tensor=True,
            **act_kwargs,
        )

        # Layer 1: quantized aggregation + linear (weight precision = bit_width)
        self.conv1 = QuantGCNConv(
            in_features, hidden,
            weight_quant=weight_quant,
            **weight_kwargs,
        )

        self.relu = nn.ReLU()

        # Layer 2: quantized aggregation + linear (weight precision = bit_width)
        self.conv2 = QuantGCNConv(
            hidden, num_classes,
            weight_quant=weight_quant,
            **weight_kwargs,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_quant(x)         # → QuantTensor
        x = self.conv1(x, edge_index)   # → Tensor  (quantized weights)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)   # → Tensor  (quantized weights)
        return x if not isinstance(x, QuantTensor) else x.value


# ── Training utilities ────────────────────────────────────────────────────────

def _train_step(model, optimizer, criterion, x, edge_index, labels, train_mask):
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index)
    loss = criterion(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _accuracy(model, x, edge_index, labels, mask) -> float:
    model.eval()
    logits = model(x, edge_index)
    return (logits[mask].argmax(1) == labels[mask]).float().mean().item()


def fit(
    model: FixedQuantGCN,
    optimizer,
    criterion,
    x: Tensor,
    edge_index: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    epochs: int = 200,
    patience: int = 50,
    verbose: bool = True,
) -> FixedQuantGCN:
    """
    Train with early stopping on validation accuracy.
    Restores the best checkpoint before returning.
    """
    best_val_acc = -1.0
    best_state: Optional[dict] = None
    patience_counter = 0

    for epoch in range(epochs):
        loss = _train_step(model, optimizer, criterion, x, edge_index, labels, train_mask)
        val_acc = _accuracy(model, x, edge_index, labels, val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 50 == 0:
            train_acc = _accuracy(model, x, edge_index, labels, train_mask)
            print(f"  [{model.bit_width}-bit] Epoch {epoch+1:4d} | "
                  f"loss={loss:.4f} | train={train_acc:.4f} | val={val_acc:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  [{model.bit_width}-bit] Early stopping at epoch {epoch + 1}  "
                      f"(best val_acc={best_val_acc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    PLANETOID = ('Cora', 'Citeseer', 'Pubmed')
    AMAZON    = ('Computers', 'Photo')
    ALL_BIT_WIDTHS = [8, 4, 2, 1]

    parser = argparse.ArgumentParser(description='Fixed-precision Brevitas GCN')
    parser.add_argument('--hidden',    type=int, default=16)
    parser.add_argument('--epochs',    type=int, default=200)
    parser.add_argument('--lr',        type=float, default=1e-2)
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--bit_width', type=int, default=None,
                        choices=[1, 2, 4, 8],
                        help='Single bit width to run. '
                             'Omit to run all four (8 → 4 → 2 → 1).')
    parser.add_argument('--output',    type=str, default='fix_result.txt',
                        help='File to save results (default: fix_result.txt)')
    args = parser.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bit_widths_to_run = [args.bit_width] if args.bit_width is not None else ALL_BIT_WIDTHS

    # all_results[dataset][bit_width] = test_acc
    all_results: dict[str, dict[int, float]] = {}

    ALL_DATASETS = list(PLANETOID) + list(AMAZON)

    for DATASET in ALL_DATASETS:
        print(f"\n{'═'*60}")
        print(f"  Dataset: {DATASET}")
        print(f"{'═'*60}")

        # ── Load data ─────────────────────────────────────────────────────────
        if DATASET in PLANETOID:
            dataset = Planetoid(root='./data', name=DATASET, split='random')
            graph   = dataset[0]
        else:
            dataset = Amazon(root='./data', name=DATASET)
            graph   = T.RandomNodeSplit(num_val=0.1, num_test=0.2)(dataset[0])

        x          = graph.x.to(device)
        labels     = graph.y.to(device)
        edge_index = graph.edge_index.to(device)
        train_mask = graph.train_mask.to(device)
        val_mask   = graph.val_mask.to(device)
        test_mask  = graph.test_mask.to(device)

        IN_FEATURES  = dataset.num_features
        NUM_CLASSES  = dataset.num_classes
        dropout      = 0.5 if DATASET in PLANETOID else 0.8
        weight_decay = 5e-4 if DATASET in PLANETOID else 1e-3

        print(f"  features={IN_FEATURES}  classes={NUM_CLASSES}  device={device}")
        print(f"  train={train_mask.sum().item()}  "
              f"val={val_mask.sum().item()}  "
              f"test={test_mask.sum().item()}")

        ds_results: dict[int, float] = {}

        for bw in bit_widths_to_run:
            print(f"\n{'─'*55}")
            print(f"  Running {bw}-bit quantization  [{DATASET}]")
            print(f"{'─'*55}")

            # Re-seed before each run for fair comparison
            torch.manual_seed(args.seed)

            model = FixedQuantGCN(
                IN_FEATURES, args.hidden, NUM_CLASSES,
                bit_width=bw, dropout=dropout,
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=weight_decay
            )
            criterion = nn.CrossEntropyLoss()

            model = fit(
                model, optimizer, criterion,
                x, edge_index, labels,
                train_mask, val_mask,
                epochs=args.epochs, patience=20, verbose=True,
            )

            test_acc = _accuracy(model, x, edge_index, labels, test_mask)
            ds_results[bw] = test_acc
            print(f"  Test accuracy  ({bw}-bit): {test_acc:.4f}")

        all_results[DATASET] = ds_results

    # ── Console summary ───────────────────────────────────────────────────────
    header = f"{'Dataset':<12}" + "".join(f"  {bw}-bit" for bw in bit_widths_to_run)
    sep    = "─" * len(header)
    print(f"\n\n{'═'*60}")
    print("  Fixed-precision GCN  –  Test accuracy summary")
    print(f"{'═'*60}")
    print(header)
    print(sep)
    for ds in ALL_DATASETS:
        row = f"{ds:<12}"
        for bw in bit_widths_to_run:
            row += f"  {all_results[ds][bw]:.4f}"
        print(row)
    print(f"{'═'*60}\n")

    # ── Save to file ──────────────────────────────────────────────────────────
    with open(args.output, 'w') as f:
        f.write("Fixed-precision Brevitas GCN  –  Test accuracy\n")
        f.write(f"Seed={args.seed}  Hidden={args.hidden}  "
                f"Epochs={args.epochs}  LR={args.lr}\n")
        f.write(f"{'─'*60}\n")
        f.write(f"{'Dataset':<12}")
        for bw in bit_widths_to_run:
            f.write(f"  {bw}-bit  ")
        f.write("\n")
        f.write(f"{'─'*60}\n")
        for ds in ALL_DATASETS:
            f.write(f"{ds:<12}")
            for bw in bit_widths_to_run:
                f.write(f"  {all_results[ds][bw]:.4f}  ")
            f.write("\n")
        f.write(f"{'─'*60}\n")

    print(f"Results saved to {args.output}")
