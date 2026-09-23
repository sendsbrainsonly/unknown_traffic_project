# CipherSpectrum MIX Provenance Summary

- MIX_STATUS = **PARTIALLY_OVERLAPPING**.
- MIX PCAPs: **40,200**.
- Exact SHA-256 duplicate MIX files: **0 (0.00%)**.
- Normalized packet-fingerprint overlap: **0 (0.00%)**.
- MIX collection groups shared with canonical sources: **16,596/18,260 (90.89%)**.
- Canonical three-cipher candidate: **120,000 PCAPs (40 classes x 3,000)**.
- Primary recommendation: **exclude MIX**. No source file was deleted or changed.

The normalized fingerprint excludes the PCAP global header and combines packet count with the first eight packets' lengths, direction, relative timestamps, and complete frame-byte hashes. Shared collection IDs alone establish collection/session dependence, not byte identity.
