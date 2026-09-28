# Experiment result index

- Stage43 completion addendum (2026-09-28): 300 frozen-score mixtures and 600,000 memberships independently replayed PASS (1,200 metrics, max error 0). DES-v1 was prevalence-stable but composition-sensitive: all-balanced AUROC/UFAR `0.980976/0.067100`, authentication/web `0.995600/0`, Bot+DDoS `0.947600/0.338350`. Final Gate `COMPOSITION_SENSITIVE`. [Full results](stage43_cic_mixed_prevalence/RESULTS.md).

- Stage43 CIC mixed-Unknown composition/prevalence stress test: running; score-only reuse of frozen Stage42-S–Y sample scores, no encoder/threshold fitting and no PCAP access. [Progress and eventual results](stage43_cic_mixed_prevalence/RESULTS.md).

- Stage42-Y completion addendum (2026-09-28): four prelisted CIC candidates completed in frozen order; all eight metric rows per candidate independently replayed PASS, Unknown/Test fitting zero. DES-v1 natural AUROC SSH/Brute/XSS/Slowhttptest = `0.991275/0.995652/0.997489/0.993400`; UFAR = `0/0/0/0.001374`; Known FRR = `0.071303`. Post-hoc diagnostic, not untouched validation. [Full results](stage42y_cic_remaining_four_open_set/RESULTS.md).

- Stage42-Y four prelisted CIC candidates: running in fixed order, with all four manifests and protocols frozen before new packet-feature access. SSH-Patator 2,987; Web Brute Force 1,364; Web XSS 629; full Slowhttptest 5,096 (132 previously evaluated). No final metrics yet. [Progress and results](stage42y_cic_remaining_four_open_set/RESULTS.md).

- Stage42-X completion addendum (2026-09-28): FTP-Patator diagnostic completed and independently replayed PASS. DES-v1 natural AUROC/AUPRC/UFAR = `0.998342/0.998596/0.000000`, but MSP/Energy/Centroid are also very strong; capture/schedule confounding remains. The earlier `running` entry below is historical. [Full results](stage42x_cic_ftp_patator_open_set/RESULTS.md).

- Stage42-X sixth favorable CIC candidate: running; frozen `BENIGN + PortScan` Known model with all 3,985 matched Tuesday `FTP-Patator` flows as Unknown Test, protocol frozen before packet-feature access. [Progress and eventual results](stage42x_cic_ftp_patator_open_set/RESULTS.md).

- Stage42-W completion addendum (2026-09-28): full Bot diagnostic independently replayed PASS, but the favorable screen failed. DES-v1 natural AUROC/AUPRC/UFAR = `0.946433/0.875758/0.508958`; true 1:1 AUROC/AUPRC = `0.950623/0.936257`. The earlier `running` entry below is historical. [Full results](stage42w_cic_bot_open_set/RESULTS.md).

- Stage42-W fifth favorable CIC candidate: running; frozen `BENIGN + PortScan` Known model with all 1,228 matched Friday `Bot` flows as Unknown Test. Candidate and exact 1:1 Known membership were frozen before packet-feature access. [Progress and eventual results](stage42w_cic_bot_open_set/RESULTS.md).

- Stage42-V completion addendum (2026-09-28): full DDoS diagnostic independently replayed PASS, but the favorable screen failed. DES-v1 natural AUROC/AUPRC/UFAR = `0.948990/0.994485/0.166134`; balanced 1:1 AUPRC = `0.882042`. The earlier `running` entry below is historical. [Full results](stage42v_cic_ddos_open_set/RESULTS.md).

- Stage42-V fourth favorable CIC candidate: running; frozen `BENIGN + PortScan` Known model with all 76,613 matched Friday `DDoS` flows as Unknown Test, protocol frozen before packet-feature access. [Progress and eventual results](stage42v_cic_ddos_open_set/RESULTS.md).

- Stage42-U completion addendum (2026-09-28): Hulk diagnostic completed and independently replayed PASS; DES-v1 natural AUROC/AUPRC/UFAR = 0.973767/0.998806/0.001134; balanced 1:1 AUPRC = 0.928132. The earlier `running` entry below is historical. [Full results](stage42u_cic_hulk_open_set/RESULTS.md).

- Stage42-U third favorable CIC candidate: running; frozen `BENIGN + PortScan` Known model with all 155,168 matched `DoS Hulk` flows as Unknown Test, protocol frozen before packet-feature access. [Progress and eventual results](stage42u_cic_hulk_open_set/RESULTS.md).

- Stage42-T second favorable CIC candidate: complete (diagnostic); frozen `BENIGN + PortScan` model, all 7,441 matched `DoS GoldenEye` flows as Unknown Test. DES-v1 natural AUROC/AUPRC/UFAR = 0.96996/0.97719/0.00242; independent replay PASS. [Results and limitations](stage42t_cic_goldeneye_open_set/RESULTS.md).

- Stage42-S first favorable CIC candidate: complete (diagnostic); frozen `BENIGN + PortScan` model with all 5,709 matched `DoS slowloris` flows as Unknown Test. DES-v1 AUROC/AUPRC = 0.99305/0.99672 (natural prevalence), with Known FRR 0.07130. Protocol was frozen before packet-feature access; independent replay PASS. [Results and limitations](stage42s_cic_favorable_open_set/RESULTS.md).

- Stage41 A-1/A-2/A-3 matched-flow Open-Detect vs three-view DES-v1: running; [results](stage41_a123_matched_open_set_comparison/RESULTS.md).
- Stage41 matched A-1/A-2/A-3 same-flow comparison: success; [results](stage41_a123_matched_open_set_comparison/RESULTS.md).
- Stage41 final reproduction note (seed 2022, Known-Val P95): balanced AUROC OD/three-view A-1 `0.9907/0.9769`, A-2 `0.9290/0.9406`, A-3 `0.9782/0.9609`; A-2 Geodo dominates false negatives and balanced F1 must be read alongside natural-prevalence F1. [Full record](stage41_a123_matched_open_set_comparison/RESULTS.md).
- Stage41 closed/open paired record: on the same Known Test flows, the three-view method has higher Accuracy, Macro-F1 and Weighted-F1 in A-1/A-2/A-3; open-set AUROC/AUPRC improve only in A-2. [Closed/open tables and limits](stage41_a123_matched_open_set_comparison/RESULTS.md).
- Stage40 CIC independent resume v1: coordinator hit the 60-second tmux wait limit (exit 124), while its child continued; [failure record](stage40_ustc_cic_open_set/cic_resume_20260927/RESULTS.md).
- Stage40 CIC sequential resume v2: complete; both frozen Unknown-PortScan and Unknown-Slowhttptest pilots independently replayed PASS. At Known-Val P95, DES-v1 AUROC/UFAR is `0.463178/0.948063` for PortScan and `0.997753/0.000000` for Slowhttptest; role-dependent single-seed diagnostic, not a general CIC claim. [Full CIC/USTC record](stage40_ustc_cic_open_set/RESULTS.md); [attempt evidence](stage40_ustc_cic_open_set/cic_resume_20260927_v2/RESULTS.md).
