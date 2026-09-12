# A-3 Open-Detect Smoke Report

- Status: **PASS**
- Unknown samples loaded: 0
- Known classes / prototypes: 15 / 15
- Prototype shape: `[15, 128]`
- Latent `mu`/`logvar` dimension: 128
- Train / validation samples: 480 / 240
- Epochs completed: 3
- Total and component losses finite: true
- NaN/Inf observed: no
- Upper-logvar stability-guard activations: 0
- Best validation composite: 0.014814815
- Physical GPU: `3`

The smoke graph exercised raw-byte images, encoder `mu`/`logvar`, stochastic
latent sampling, 15 learned prototypes, decoder reconstruction, and
all released loss terms. No held-out Unknown sample was fetched.
