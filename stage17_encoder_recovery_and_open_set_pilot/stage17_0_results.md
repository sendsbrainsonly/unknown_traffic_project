# Stage 17-0 results

The historical pipeline was found. Its actual branches are a 768-D TrafficFormer, a 128-D per-flow FIG/TAGCN, and an 896-D train-standardized concatenation with a linear probe. Model A and B have complete USTC-20 checkpoints; Model C has source code but no saved historical checkpoint. None of the historical A/B/C checkpoints is a valid five-Known Service-LOSO checkpoint. The recovered architecture and feature code can run locally and is therefore eligible for protocol-specific retraining.
