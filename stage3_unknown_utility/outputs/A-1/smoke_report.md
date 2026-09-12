# A-1 Open-Detect Smoke Report

- Status: **PASS**
- Unknown samples loaded: 0
- Known classes / prototypes: 19 / 19
- Prototype shape: `[19, 128]`
- Latent `mu`/`logvar` dimension: 128
- Train / validation samples: 608 / 304
- Epochs completed: 3
- Total and component losses finite: true
- NaN/Inf observed: no
- Best validation composite: 0.009569378
- Physical GPU: `4`

The smoke graph exercised raw-byte images, encoder `mu`/`logvar`, stochastic
latent sampling, 19 learned prototypes, decoder reconstruction, and
all released loss terms. No held-out Unknown sample was fetched.
