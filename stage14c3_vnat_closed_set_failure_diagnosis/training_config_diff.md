# Open-Detect native versus F0/F2 training configuration

All values below were read from the executed configs and source files. “Same”
means the effective value is equal, not merely that the implementation is
similar.

| Item | Native Open-Detect | F0 | F2 | Material difference |
|---|---|---|---|---|
| Model / encoder | Released `OpenDetectNet`, ResNet-18 VAE | `Stage3OpenDetectNet` wrapping the same released encoder/decoder | Same as F0 | Same network graph except F0/F2 logvar upper clamp |
| Input shape / dimension | 1x32x32, 1024 bytes, latent 128 | Same byte image and latent 128 | Same 1024-byte image; 64 zero address slots replaced by 8 robust-quantized flow statistics | F2 changes 6.25% of input positions; does not add dimensions |
| Preprocessing | First 8 packets; 80 header + 48 payload; masked IPv4 addresses; divide by 255 once | Byte-identical F0 arrays; divide by 255 once | F0 plus log/log-ratio -> train median/IQR -> clip [-4,4] -> uint8 [0,255], then divide by 255 once | F2 scaler is additional but bounded |
| Scaler | None | None | Fit on Known Train only and reused unchanged for Train/Val | Independently recomputed in `feature_distribution_audit.csv` |
| Train augmentation | PIL RandomCrop(32,padding=4) + RandomHorizontalFlip | Tensor versions of the same transforms | Same as F0 | Parity audit: 0/100 mismatches; max diff 0.0 |
| Loss | `0.005*(rec+kld+ent)+0.995*dis` | Same | Same | None |
| Class weight | None | None | None | All losses are sample-weighted by observed class frequency |
| Sampler | RandomSampler (`shuffle=True`) | Same | Same | None; no weighted sampler |
| Batch size | 128 | 512 | 512 | F0/F2 have about one quarter as many optimizer steps per epoch |
| DataLoader workers | 4 | 0 | 0 | Runtime/augmentation RNG difference; not a loss change |
| Optimizer | Adam, lr=0.001, betas=(0.9,0.999), weight_decay=0 | Same | Same | None |
| Scheduler | MultiStepLR milestones [50,80], gamma=0.1 | Same configured | Same configured | F0/F2 stop before epoch 30, so milestones never execute |
| Maximum epochs | 100, always completed | 100 maximum | 100 maximum | Effective epochs differ because of early stopping |
| Gradient clipping | None | None | None | None |
| Initialization | Released Xavier/He/BatchNorm init; Kaiming prototypes | Same released initializer and prototype init | Same | None |
| Random seed | Fixed training seed 2022 for every protocol | Protocol seed 2022–2026 | Protocol seed 2022–2026 | Material stochastic-policy difference |
| Dropout | None in released ResNet/decoder | None | None | None |
| Checkpoint selection | Maximum Known-Val Accuracy | Maximum harmonic mean of Known-Val Accuracy and Macro-F1 | Same as F0 | Material selection difference |
| Early stopping | None; 100 epochs | Patience 5 on validation composite | Same as F0 | Material: all F0/F2 runs stop at epochs 9–30 |
| Prototype reset | After zero-based epochs 50 and 80 | Same configured | Same configured | Never reached by any frozen F0/F2 run |
| Numerical protection | Released raw logvar | Clamp raw logvar upper tail at 20 | Same as F0 | Guard triggered only once across F0 and zero times across F2 |
| Label encoding | Frozen Known-class order, contiguous local IDs 0..C-1 | Same | Same | Hash/equality audit PASS |

## Direct configuration implications

1. Optimizer, learning rate, weight decay, scheduler definition, loss,
   initialization, class weighting and sampling are not configuration
   explanations by themselves because they are equal.
2. The scheduler and both prototype resets are *configured* in F0/F2 but are
   operationally dead: patience-5 stopping ends every run before epoch 31.
3. Batch 512 further reduces update count by approximately 4x per epoch. The
   combination of early stopping and batch size creates a much larger update
   budget gap than the nominal “100 epochs” fields suggest.
4. Numerical protection is unlikely to explain the aggregate gap because its
   activation counter is zero for 29/30 F0/F2 runs and one batch for only
   F0 Medium-2025.
