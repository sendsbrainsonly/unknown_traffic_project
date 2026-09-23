# DQ-4 TrafficFormer Eligibility Audit

- Existing processing audit rows: 17769
- Successful parent flows: 1526
- Final Service manifest rows: 1526
- Status counts: `{"filtered:size_lt_2kb": 16243, "success": 1526}`
- Success-parent to final-Service multiset parity: `True`
- Actual code order: reject captured bytes `<2048`; then reject packet count `<3`; then encode first 5 packets x 64 bytes.
- C_PARENT is the Native session whose capture-wide bidirectional five-tuple parent passes that rule.
- C_FINAL equals C_PARENT for the six-Service task because every successful Service class has >=10 parents and the final Service manifest contains the complete success multiset.
- Limitation: TrafficFormer parent flow and Native 60-second/TCP-boundary session are not one-to-one; C is a parent-derived selection subset, not a claim that each Native row equals one final TrafficFormer row.
