"""
The encoder block from the Lucidchart spec.

Every hyperparameter below is read off the flowchart, not chosen by me:

  flowchart box                        ->  code
  ------------------------------------------------------------------------
  12 attention heads, Q/K/V are 4x64   ->  nhead=12 with d_model=768 (768/12 = 64)
  scale by sqrt(d_k), d_k = 64         ->  built into nn.MultiheadAttention
  concat 12 heads x W^O (768 x 768)    ->  the layer's internal out_proj
  "add ... then normalization happens" ->  norm_first=False  (post-LN, as in Vaswani 2017)
  epsilon = 10^-6                      ->  layer_norm_eps=1e-6
  768 -> 3072, ReLU, 3072 -> 768       ->  dim_feedforward=3072, activation='relu'
  "encoder block repeats 6 times"      ->  num_layers=6, independent weights per layer

Dropout 0.1 and independent per-layer weights come from Vaswani et al. 2017
(sections 5.4 and 3.1) -- the flowchart is silent on both.

Head: mean-pool the 4 token rows -> Linear(768, 3). No positional encoding: the
metric identity is carried by the embedded template text.
"""

import torch
import torch.nn as nn

D_MODEL = 768
N_HEADS = 12
D_FF = 3072
N_LAYERS = 6
LAYER_NORM_EPS = 1e-6
DROPOUT = 0.1
N_CLASSES = 3
SEQ_LEN = 4

CLASSES = ["bearish", "neutral", "bullish"]


class NumericChannel(nn.Module):
    """
    The separate numerical path (Option C).

    The text template tells the model WHICH metric a row is; measurement showed it
    cannot carry HOW BIG the number is (0.0031 cos-distance across five orders of
    magnitude). So the magnitude arrives on its own channel: each of the 4 metrics
    gets its own learned projection of its scalar into d_model,

        numeric_j = s_j * W_j + b_j,      W_j, b_j in R^768

    which is the numerical feature-tokenizer used for numeric features in
    transformer models for tabular data (FT-Transformer, Gorishniy et al. 2021).
    The result is summed into the text embedding, exactly as the 2017 paper sums
    positional encodings into token embeddings -- so the 4 x 768 shape entering
    the encoder block is unchanged and the flowchart still holds downstream.
    """

    def __init__(self, seq_len=SEQ_LEN, d_model=D_MODEL):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(seq_len, d_model))
        self.bias = nn.Parameter(torch.zeros(seq_len, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, s):
        # s: (B, 4) standardised scalars -> (B, 4, 768)
        return s.unsqueeze(-1) * self.weight + self.bias


class EarningsEncoder(nn.Module):
    def __init__(self, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.numeric = NumericChannel()
        # Each layer is constructed separately. nn.TransformerEncoder would clone a
        # single prototype with copy.deepcopy, which leaves all 6 layers starting
        # from *identical* weights; Vaswani 2017 sec 3.1 stacks N layers that are
        # identical in structure, each independently initialised.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=N_HEADS,
                dim_feedforward=D_FF,
                dropout=dropout,
                activation="relu",
                layer_norm_eps=LAYER_NORM_EPS,
                norm_first=False,      # post-LN: add THEN normalise
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(D_MODEL, N_CLASSES)

    def build_input(self, text, scal):
        """
        text (B,4,768) + scalars (B,4) -> the 4 x 768 matrix the chart feeds in.

        Scaled by sqrt(d_model), per Vaswani sec 3.4 ("in the embedding layers, we
        multiply those weights by sqrt(d_model)"). Measured justification: layers 1-5
        receive LayerNorm output, whose row norm is sqrt(768) = 27.71 by construction,
        and their attention differentiates freely (0.15-0.33 deviation from uniform).
        Layer 0 alone saw the raw embedding at norm 1.32 -- 21x smaller -- which
        squashed QK^T/sqrt(d_k) toward zero and pinned its softmax at uniform (0.0043).
        The scaling is applied to the summed text+numeric matrix so the relative
        weighting of the two channels, which the input gate validated, is preserved.
        """
        return (text + self.numeric(scal)) * (D_MODEL ** 0.5)

    def forward(self, text, scal, assert_shapes=False):
        x = self.build_input(text, scal)
        if assert_shapes:
            assert x.dim() == 3 and x.shape[1:] == (SEQ_LEN, D_MODEL), \
                f"expected (B, {SEQ_LEN}, {D_MODEL}), got {tuple(x.shape)}"
        h = x
        for layer in self.layers:
            h = layer(h)
            if assert_shapes:
                assert h.shape == x.shape, f"layer changed shape: {tuple(h.shape)}"
        if assert_shapes:
            assert h.shape == x.shape, f"encoder changed shape: {tuple(h.shape)}"
        pooled = h.mean(dim=1)
        if assert_shapes:
            assert pooled.shape == (x.shape[0], D_MODEL), tuple(pooled.shape)
        logits = self.head(pooled)
        if assert_shapes:
            assert logits.shape == (x.shape[0], N_CLASSES), tuple(logits.shape)
        return logits

    @torch.no_grad()
    def attention_at(self, text, scal, idx=0):
        """
        The 4x4 attention weight matrix at layer `idx`, averaged over the 12 heads.
        Layers before `idx` are run first so the probe sees that layer's real input.
        """
        was_training = self.training
        self.eval()
        h = self.build_input(text, scal)
        for layer in self.layers[:idx]:
            h = layer(h)
        sa = self.layers[idx].self_attn
        _, w = sa(h, h, h, need_weights=True, average_attn_weights=True)
        if was_training:
            self.train()
        return w

    def first_layer_attention(self, text, scal):
        """Layer-0 attention. Kept for the demo in predict.py."""
        return self.attention_at(text, scal, 0)

    @torch.no_grad()
    def attention_uniformity(self, text, scal, idx=0):
        """max |attn - 1/seq_len|. 0 means perfectly uniform (pure averaging)."""
        w = self.attention_at(text, scal, idx)
        return float((w - 1.0 / w.shape[-1]).abs().max())


def param_count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def self_test():
    """Shape assertions against every stage of the flowchart."""
    torch.manual_seed(0)
    m = EarningsEncoder()
    text = torch.randn(5, SEQ_LEN, D_MODEL)
    scal = torch.randn(5, SEQ_LEN)
    x = m.build_input(text, scal)

    logits = m(text, scal, assert_shapes=True)
    attn = m.first_layer_attention(text, scal)

    layer0 = m.layers[0]
    checks = [
        ("input matrix", tuple(x.shape[1:]), (4, 768)),
        ("attention weights (per chart: 4 x 4)", tuple(attn.shape[1:]), (4, 4)),
        ("W^O projection", tuple(layer0.self_attn.out_proj.weight.shape), (768, 768)),
        ("FFN W_1", tuple(layer0.linear1.weight.shape), (3072, 768)),
        ("FFN W_2", tuple(layer0.linear2.weight.shape), (768, 3072)),
        ("output logits", tuple(logits.shape[1:]), (3,)),
    ]
    ok = True
    print("  shape checks against the flowchart:")
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(f"    {'OK ' if good else 'FAIL'}  {name:38s} {got} == {want}")

    l0, l1 = m.layers[0], m.layers[1]
    separate_objs = l0.linear1.weight is not l1.linear1.weight
    separate_vals = not torch.equal(l0.linear1.weight, l1.linear1.weight)
    independent = separate_objs and separate_vals
    print(f"    {'OK ' if independent else 'FAIL'}  "
          f"{'6 independently-init layers':38s} "
          f"distinct-tensors={separate_objs} distinct-values={separate_vals}")
    ok &= independent

    n_layers_ok = len(m.layers) == 6
    print(f"    {'OK ' if n_layers_ok else 'FAIL'}  {'encoder repeats 6 times':38s} "
          f"{len(m.layers)} layers")
    ok &= n_layers_ok

    nz = m.numeric(torch.tensor([[0., 0., 0., 0.], [3., 3., 3., 3.]]))
    numeric_live = float((nz[1] - nz[0]).abs().mean()) > 1e-6
    print(f"    {'OK ' if numeric_live else 'FAIL'}  "
          f"{'numeric channel responds to scalars':38s} "
          f"delta={float((nz[1]-nz[0]).abs().mean()):.4f}")
    ok &= numeric_live

    per_metric = not torch.equal(m.numeric.weight[0], m.numeric.weight[1])
    print(f"    {'OK ' if per_metric else 'FAIL'}  "
          f"{'per-metric numeric projections':38s} {per_metric}")
    ok &= per_metric

    eps_ok = layer0.norm1.eps == LAYER_NORM_EPS
    print(f"    {'OK ' if eps_ok else 'FAIL'}  {'layer-norm eps == 1e-6':38s} "
          f"{layer0.norm1.eps}")
    ok &= eps_ok

    postln = layer0.norm_first is False
    print(f"    {'OK ' if postln else 'FAIL'}  {'post-LN (add then normalise)':38s} "
          f"norm_first={layer0.norm_first}")
    ok &= postln

    heads_ok = layer0.self_attn.num_heads == 12 and D_MODEL // 12 == 64
    print(f"    {'OK ' if heads_ok else 'FAIL'}  {'12 heads, d_k = 64':38s} "
          f"{layer0.self_attn.num_heads} heads")
    ok &= heads_ok

    print(f"\n  trainable parameters: {param_count(m):,}")
    print(f"  self-test: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
