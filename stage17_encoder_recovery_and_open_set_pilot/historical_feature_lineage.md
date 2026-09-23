# Historical feature lineage

- **TrafficFormer (G1/G2/G8):** first five observed packets; strip 14-byte Ethernet header; retain up to 64 following bytes per packet; overlapping two-byte bigrams; prepend `[SEP]` per packet; downstream token tail padding to 320. It does not explicitly input length, direction, IAT, burst statistics, or full-flow statistics.
- **FIG/TAGCN (G3/G4/G5):** at most 30 packet nodes. Seven features are direction, captured length, time from first packet, current-burst packet count, current-burst byte count, current/previous burst packet ratio, and current/previous burst byte ratio. Bursts are maximal same-direction runs. Edges join consecutive packets inside bursts and first-to-first/last-to-last nodes of adjacent bursts.
- **Fusion:** no new raw feature. Each branch is standardized with Known-Train statistics and concatenated.
- **Not used:** full-flow aggregate statistics, explicit transport-state features, capture ID, filename, labels, or quality mask as learned inputs. Padding masks are used only by batching/readout.

The recovered graph branch already covers direction, packet timing, length and burst behavior; a future Behavior branch would need non-duplicative evidence rather than relabeling these same features.
