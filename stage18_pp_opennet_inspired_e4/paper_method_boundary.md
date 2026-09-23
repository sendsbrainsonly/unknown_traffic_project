# PP-OpenNet evidence and implementation boundary

## Paper-confirmed design

The local INFOCOM 2026 paper explicitly uses packet timestamps and lengths only, a multi-branch 1-D CNN with kernel sizes 1/3/5 plus max-pooling, a temporal module, GRU-based recurrent feature fusion, ARPL reciprocal points and a training-time stochastic projection. Its paper experiment uses 1,000-packet, two-second retroactive slices, tsfresh statistics, AdamW at 0.001 for 400 epochs, and a background-traffic training class.

## Unavailable official artifact

No official repository was linked by the paper, the author publication page, the public author GitHub account, or the Huawei Theory Lab organization as checked on 2026-09-20. Therefore Stage 18 is not an author-code reproduction.

## Project-autonomous E4 decisions

Stage 18 independently implements masked packet-sequence convolutions, a packet GRU, branch-level GRU fusion, a 128-D embedding, a linear Known-class head, inverse-sqrt-frequency weighted cross entropy, Known-Train robust feature normalization, and MSP/Energy/empirical-centroid scores. These details are not attributed to the paper. ARPL/PVRP and background-traffic training are intentionally excluded because they would change the frozen Strict Unknown-Free protocol and cannot be reconstructed exactly from the paper.

## Claim boundary

The valid claim is: a controlled, PP-OpenNet-inspired independent encoder benchmark on the frozen Stage 17 30-packet Service-LOSO inputs. It is not PP-OpenNet paper-equivalent reproduction, not an evaluation on the paper's composite 49.66k sequence dataset, and not untouched external validation.
