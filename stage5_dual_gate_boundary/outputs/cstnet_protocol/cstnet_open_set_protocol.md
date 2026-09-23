# CSTNET-TLS1.3 Strict Unknown-Free Protocol Freeze

## Scope

This document freezes class eligibility, group-aware Known splits and three
class-held-out folds. No CSTNET encoder training, representation export,
density fitting, threshold calibration or Unknown scoring was run.

## Dataset and Eligibility

- Original classes: `120`
- Readable per-flow PCAPs: `46372`
- Engineering eligibility rule fixed before modeling: `sample_count >= 100`
- Eligible/excluded classes: `119/1`
- Excluded: `chia.net`
- Inventory fingerprint: `908cd511c40f74e7daaca9829dddd9ba92e652daafa8bedc572702008f65b476`

## Group-Aware Split

The source has no authoritative capture/session manifest. `group_id` is frozen
as the UTC one-minute window containing the first packet timestamp. Before
Known splitting, Known samples sharing any group with a selected Unknown class
are excluded. Remaining Known groups are assigned by a deterministic shuffled
10-fold `StratifiedGroupKFold`; two folds are selected for Validation and Test
using the frozen count-only lexicographic criterion, and eight folds form Train.

This makes active Train/Validation/Test/Unknown groups disjoint. It is a proxy,
not proof of true session identity.

## Domain and Endpoint Risk

`PROTOCOL_RISK`: the domain directory is also the class label. Endpoint metadata
was not flow-parsed by the source audit, so endpoint novelty remains unresolved.
The capture-window control does not eliminate that semantic shortcut risk.

## Frozen Folds

### Low- Seed: `42`- Known/Unknown classes: `113/6`- Unknown classes: `ampproject.org, ibm.com, overleaf.com, spring.io, unity3d.com, yy.com`- Selection attempt: `1`; rejected attempts: `0`- Known train/validation/test samples: `34550/4337/4236`- Purged Known samples sharing an Unknown capture group: `962`### Medium- Seed: `43`- Known/Unknown classes: `101/18`- Unknown classes: `51cto.com, adobe.com, codepen.io, dailymotion.com, deepl.com, duckduckgo.com, gmail.com, grammarly.com, hubspot.com, ibm.com, mi.com, naver.com, nike.com, smzdm.com, squarespace.com, teads.tv, weibo.com, wikimedia.org`- Selection attempt: `1`; rejected attempts: `0`- Known train/validation/test samples: `30288/3777/3798`- Purged Known samples sharing an Unknown capture group: `1981`### High- Seed: `44`- Known/Unknown classes: `89/30`- Unknown classes: `alipay.com, asus.com, atlassian.net, bilibili.com, cloudflare.com, dailymotion.com, deepl.com, digitaloceanspaces.com, eastmoney.com, ggpht.com, google.com, grammarly.com, ieee.org, instagram.com, iqiyi.com, media.net, nike.com, onlinedown.net, oracle.com, qq.com, semanticscholar.org, snapchat.com, sohu.com, squarespace.com, steampowered.com, tiktok.com, vmware.com, xiaomi.com, ximalaya.com, yahoo.com`- Selection attempt: `4`; rejected attempts: `3`- Known train/validation/test samples: `25710/3221/3227`- Purged Known samples sharing an Unknown capture group: `3322`
## Strict Unknown-Free Boundary

Unknown classes are forbidden from encoder training/validation, checkpoint
selection, scaler/PCA/Gaussian/GMM fitting, boundary calibration and tuning.
They may be accessed only in the future final test, after this freeze.

`created_before_unknown_evaluation = true`
