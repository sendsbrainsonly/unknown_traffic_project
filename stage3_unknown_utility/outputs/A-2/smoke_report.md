# A-2 Open-Detect Smoke Report

- Status: **PASS**
- Unknown samples loaded: 0
- Known classes / prototypes: 17 / 17
- Prototype shape: `[17, 128]`
- Latent `mu`/`logvar` dimension: 128
- Train / validation samples: 544 / 272
- Epochs completed: 3
- Total and component losses finite: true
- NaN/Inf observed: no
- Best validation composite: 0.049952477
- Physical GPU: `4`

The smoke graph exercised raw-byte images, encoder `mu`/`logvar`, stochastic
latent sampling, 17 learned prototypes, decoder reconstruction, and
all released loss terms. No held-out Unknown sample was fetched.
