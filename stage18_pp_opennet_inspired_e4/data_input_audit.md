# Stage 18 packet input audit

- Source cache: `stage17_encoder_recovery_and_open_set_pilot/feature_cache/service3065_historical_inputs.npz`.
- Frozen cache SHA256: `aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff`.
- Flow count: 3,065; packet count per flow: 1--30.
- `fig_x[:,:,0]`: direction sign, +1 initiator and -1 responder.
- `fig_x[:,:,1]`: captured frame length in bytes.
- `fig_x[:,:,2]`: seconds from the first packet, obtained from original packet timestamps.
- Stage 18 IAT: non-negative difference of consecutive valid relative timestamps; first packet IAT is zero.
- Padding mask: `fig_mask`; padded values are not used for normalization, pooling, recurrent length, support, or loss.
- Session identity and flow mapping are inherited unchanged from the Stage 12/17 bidirectional TCP/UDP state machine.
- The cache contains real packet observations but caps them at 30. No claim is made that this equals PP-OpenNet's 1,000-packet retroactive slices.
- Frozen roles are read from `stage16s_service_open_set_benchmark/service_unknown_protocol_manifest.csv`; Stage 18 creates no new split.
