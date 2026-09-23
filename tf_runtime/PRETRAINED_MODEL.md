# TrafficFormer official pretrained model

The model file is intentionally local-only and ignored by Git.

- Official source page: `tf_runtime/code/README.md`
- Official download URL: <https://drive.google.com/file/d/1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7/view?usp=sharing>
- Google Drive file ID: `1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7`
- Downloaded UTC date: 2026-09-11
- Download archive size: 663,982,673 bytes
- Download archive SHA-256: `a288742cff87744d7329565a9e67f131c1b4efff59d0ea02309a27748844b299`
- Archive member: `nomoe_bertflow_pre-trained_model.bin-120000`
- Extracted local path: `tf_runtime/code/models/pretrained_model.bin`
- Extracted size: 715,546,509 bytes
- Extracted SHA-256: `be9dcc1e0c6b68db005e506adf13a2d937a2f3982539a2c8a58f78c55af0d5ce`

Validation performed in the required TrafficClassifier Conda environment:

1. The official ZIP passed `unzip -t` with no errors.
2. `torch.load(..., map_location="cpu", weights_only=True)` returned an
   `OrderedDict` containing 207 tensor values.
3. The repository Model A `EncoderOnly` configuration was instantiated with
   hidden size 768, 12 Transformer layers, and the 60,005-token encrypted
   vocabulary.
4. Loading with `strict=False` produced zero missing encoder/embedding keys.
   The ten unused keys belong only to the MLM/MSP pretraining target heads,
   which are intentionally absent from `EncoderOnly`.

Re-download command (run through the workspace tmux helper in this workspace):

```bash
gdown 'https://drive.google.com/uc?id=1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7' \
  -O tf_runtime/code/models/trafficformer_pretrained_download.zip
```

The downloaded object is a ZIP archive despite the README describing it as a
pretrain model. Extract only the member named above, verify the recorded hashes,
and save that member as `pretrained_model.bin`. Do not rename the ZIP itself to
`.bin`.
