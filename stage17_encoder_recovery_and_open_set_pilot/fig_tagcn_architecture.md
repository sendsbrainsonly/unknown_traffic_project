# FIG/TAGCN architecture

Each graph represents one flow only; there are no cross-flow or cross-capture edges. `TAGCN` computes `sum_{k=0..2} Ahat^k X W_k + b`, followed by ReLU, dropout 0.5, masked mean graph readout `z_g` (128), and a linear class head. The node/edge construction is label-free and inductive.
