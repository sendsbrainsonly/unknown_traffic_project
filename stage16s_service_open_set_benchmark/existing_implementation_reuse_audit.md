# Existing implementation reuse audit

- Corrected Open-Detect trainer: `cc0efd8107fa8010415d4f64e6779c2f238c87a835db445b5d5fe55e5ccef89e`
- Corrected model: `b904447ae41a3144e2408187aa200b0796643cffb1c03ea3402dd9718820dfbd`
- Stage 14D DES implementation: `ef93184dfa9102f24e5b0a97aa2c324dc62c6621b4728cf3e653dbfd251ade25`
- Stage 15B H1 implementation: `9787bad49b88ca730a10d139ab2f2565627e4011739f8284ce2d64add72ddf95`

The five-Known-Service model is newly trained from random initialization for
each LOSO protocol/seed because every existing six-Service checkpoint has seen
the held-out Service. No six-Service checkpoint is used as initialization.
