# Metric Definition Audit

## Paper evidence

The paper reports Accuracy and F1 in closed-world and open-world tables, but it does not provide an F1 equation, averaging mode, or positive-class convention.  Therefore the exact paper definition cannot be recovered from the PDF alone.

## Released code evidence

- Closed-set classification uses `f1_score(..., average='weighted')` (`test.py:28-44`).
- Open-set evaluation forms a binary target with Known=0 and Unknown=1 and calls `f1_score(y_true, y_pred)` without an `average` argument (`test.py:46-58`).  This is binary F1 with **Unknown as the positive class**.
- The same function selects its threshold from labeled Known+Unknown test samples, so the released open-set F1 is coupled to oracle test calibration.

## Audit conclusion

- **Paper closed-world F1:** likely weighted multiclass F1, based on released code and the table pattern; confidence MEDIUM.
- **Paper open-world F1:** likely binary Known-vs-Unknown F1 with Unknown positive; confidence MEDIUM.  The PDF itself remains under-specified.
- **Stage9 Known Macro-F1:** macro-average classification F1 over accepted/reported Known classes, not the paper open-world F1.

Accordingly, paper open-world F1 and Stage9 Known Macro-F1 are **NOT_COMPARABLE**.  No claim in this audit treats them as the same metric.
