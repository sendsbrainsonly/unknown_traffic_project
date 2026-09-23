# Fusion architecture

Recovered Model C independently standardizes `z_t` and `z_g` using Known-Train means/stds, concatenates them as `z_f=[z_t;z_g]` (768+128=896), and trains a linear classifier with Adam 1e-3 for 30 epochs, selecting by Known-Validation Macro-F1. It is not end-to-end, attention-based, weighted, VAE-based, or test-tuned. No historical Model C checkpoint was found.
