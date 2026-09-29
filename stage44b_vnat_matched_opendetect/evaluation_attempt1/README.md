# Preserved first-pass evaluation

The initial direct uint8-to-Tensor adapter differed from Native PIL/ToTensor on one remote_access validation prediction. The original output is retained unaltered for diagnosis; formal metrics are recomputed under `evaluation/`.
