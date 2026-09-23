# Stage 19 command-line progress monitoring

The queue writes a self-contained status file every 15 seconds. From the project root, inspect it with:

```bash
watch -n 30 'cat stage19_ronetc_four_dataset_comparison/progress.txt'
```

For a one-time snapshot:

```bash
cat stage19_ronetc_four_dataset_comparison/progress.txt
```

The display reports total completed epochs out of 500, every task's state, epoch, assigned physical GPU, and return code. A completed epoch count is only training progress; formal Test metrics are produced only after a run finishes all 100 epochs.
