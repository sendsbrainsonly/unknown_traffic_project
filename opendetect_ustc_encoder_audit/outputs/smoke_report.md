# Open-Detect Smoke Report

- Status: PASS
- Classes represented: 20
- Train samples: 5120
- Validation samples: 1280
- Epochs: 3
- Prototype shape: `[20,128]`
- Latent dimension: 128
- First train total loss: 24.202363300
- Last train total loss: 3.910558462
- Loss decreased: True
- NaN/Inf: none observed
- Best validation accuracy: 0.050000000
- Best epoch: 1
- Physical GPU: 1

The exercised graph included image loading and augmentation, encoder `mu` and
`logvar`, stochastic reparameterized `z`, 20 prototypes, decoder
reconstruction, MSE, target-prior KL/generative loss, released discriminative
loss, entropy term, and total objective.
