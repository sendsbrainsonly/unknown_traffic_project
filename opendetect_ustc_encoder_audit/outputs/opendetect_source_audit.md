# Open-Detect Source Audit

## Scope and provenance

- Read-only upstream root: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/Open-Detect`
- Author-code checkout: `Open-Detect/code`
- Author-code commit: `b50a18515a01799468c10f6c9b60c01f8a6a4e7c`
- Root wrapper commit: `f26ac911be4324110a6632912388cd19413353bd`
- Primary implementation files: `code/model.py`, `code/networks/resnet.py`, `code/train.py`, `code/utils.py`, `code/data/dataset.py`, and `code/data/Preprocessing/{pcap2png.py,utils.py}`.
- The upstream tree is reference-only. No file in it is modified by this audit.

## Actual preprocessing

The released packet encoder converts one bidirectional flow to a one-channel
`32 x 32` byte image:

1. use at most the first 8 packets;
2. require an IPv4 layer and remove the Ethernet header by serializing from the
   IP layer;
3. set IPv4 source and destination addresses to `0.0.0.0`;
4. obtain the application `Raw` bytes when available and remove that hex string
   from the serialized IP packet to form the header portion;
5. truncate or zero-pad the header to 80 bytes and the payload to 48 bytes;
6. concatenate 128 bytes per packet, then zero-pad the flow to 1024 bytes;
7. interpret the 1024 byte values as grayscale pixels in a `32 x 32` image.

The released `read_pcap_list` uses `scapy.rdpcap`, so it loads each already
split flow PCAP in full. Its augmentation branch contains a tuple/list bug and
is irrelevant here. The main project already preserves raw Ethernet frames for
every historical Stage-0 flow, so an isolated adapter can apply the same
per-packet transform without changing the fixed flow universe.

The official processed USTC NPZ package cannot be used for this controlled
experiment: it contains only `data` and `target`, uses a different class-ID
order and sample universe, and provides no Stage-1 `flow_id`, source-flow, fold,
or seed metadata. It therefore cannot be mapped one-to-one to the historical
489,101 Stage-1 flows.

## Actual representation model

### Encoder and latent variables

- `code/networks/resnet.py::ResNet18Enc` is a custom one-channel ResNet18-like
  encoder with residual block counts `[2,2,2,2]`, a 3x3 stem, adaptive average
  pooling, and independent linear heads for `mu` and `logvar`.
- The released USTC/default hidden dimension is **128** (`train.py --h`, paper
  `d=128`).
- `mu_x` is created at `ResNet18Enc.forward`, line 121, and returned with
  `logvar` and the four lateral feature maps at line 129.
- The stochastic sample is created in `OpenDetectNet.sampler`: during training,
  `z = mu + exp(0.5*logvar)*epsilon`; in evaluation mode, `z = mu`.
- The released `OpenDetectNet.forward` returns sampled/evaluation `latent_z`,
  distances, KL values, and reconstruction, but does not expose `mu` or
  `logvar`. The isolated adapter must expose them without changing their
  computation so that the requested deterministic `mu_x` can be exported.

### Decoder

The decoder mirrors the residual stages and uses high-dropout lateral
connections from encoder feature maps. It maps latent `z` through a linear
layer, upsamples to the encoder bottleneck resolution, adds lateral features,
and reconstructs a sigmoid one-channel `32 x 32` image.

### Gaussian prototypes

- There is exactly one prototype mean per known class.
- Prototype means are trainable `nn.Parameter`s, initialized with Kaiming
  normal values.
- Prototype covariance is not learned: every class prior is
  `N(prototype_y, I)`.
- In the requested closed-set run, the prototype tensor must therefore have
  shape `[20,128]`.

## Actual released losses

For posterior `q(z|x)=N(mu_x,diag(exp(logvar_x)))` and class prior
`p(z|y)=N(prototype_y,I)`, `code/model.py` computes

`KL_y = 0.5 * (||mu_x-prototype_y||^2 + sum(exp(logvar_x)-logvar_x-1))`.

The reconstruction term is mean squared error between the sigmoid decoder
output and the input image.

The released discriminative term uses squared Euclidean distance on the
sampled latent vector, not the KL distance in paper Equation 18:

`mean(logsumexp(-dist_all/temp_inter) - logsumexp(-dist_target/temp_inter))`.

With one prototype per class, the target `logsumexp` is simply the negative
target distance. This is a nearest-prototype softmax cross-entropy form.

The released total minimized loss is

`lambda * (reconstruction + selected_target_KL + entropy) + (1-lambda) * discriminative`,

with defaults `lambda=0.005`, `temp_inter=1`, and `temp_intra=1` in
`train.py`.

## Paper/code differences and implementation hazards

1. Paper Equation 18 defines class probability with KL distance between the
   encoded Gaussian and every prototype Gaussian. Released code instead uses
   squared Euclidean distance between sampled `z` and prototype means for its
   discriminative term.
2. Released code selects the single ground-truth `dist_y` before applying the
   intra-class softmax. Its tensor has shape `[batch,1]`, so
   `softmax(-dist_y/temp_intra)` is identically one. Consequently the entropy
   term is the constant `log(num_classes)`, and the generative KL term is the
   unweighted ground-truth KL. This is released-code behavior, not the
   multi-prototype posterior described by the notation.
3. Decoder line 172 computes `x = layer1(x1)`, but line 173 reconstructs from
   `conv1(x1)` rather than `conv1(x)`. The final residual decoder stage is
   therefore calculated but omitted from the reconstruction graph.
4. `reset_prototype` replaces `model.prototypes` with a new `nn.Parameter` at
   epochs 50 and 80 after the optimizer was created. The optimizer keeps the
   old parameter reference, so reset prototypes are not subsequently updated
   by Adam unless the optimizer is repaired. This audit preserves the actual
   released reset semantics and records it as a limitation; it does not claim
   paper-equivalent prototype optimization.
5. The author code hard-codes `.cuda()` in model construction, loops and
   prototype reset. The isolated adapter makes device placement explicit; this
   is an engineering portability change, not a loss-definition change.
6. The author saves a pickled model object whenever validation accuracy exceeds
   a strictly positive initial best score. The audit saves a `state_dict` plus
   config and metrics, using only the fixed validation split for selection.
7. The author USTC loader calls its released `USTC_1c_train/test.npz` arrays and
   applies RandomCrop plus RandomHorizontalFlip in training. Those arrays are
   not the fixed historical sample universe and lack flow IDs, so only the
   transforms and byte-image semantics can be reused.

## Direct-reuse verdict

The **author network architecture and released loss semantics are reusable**
through an isolated, device-safe adapter. The author USTC NPZ data and split are
**not directly reusable** because they cannot be aligned to the original
Stage-1 flow IDs. Formal input must be reconstructed from the main project's
existing Stage-0 PKLs and joined to the immutable Stage-1 split by `flow_id`.

