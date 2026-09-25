# UBS Swiss AI Weeks 2026 – Next recurring merchant family

Predicts, per client, the merchant family of the first recurring payment after 2026-01-01
(cloud, gym, insurance, mobile, music, software, streaming, or none).

## Run
```bash
pip install -r requirements.txt
python solution.py --data-dir data --out submission.csv   # ~8 min on 2 CPU cores
```
`data/` must contain the challenge files (or `dataset.zip`, extracted automatically).

## Approach
1. **Tag transactions** with family evidence: description keywords and MCC.
2. **Detect subscription streams** with rules: cluster payments by amount, check the billing schedule,
   attach charges with noisy description/MCC that fit the schedule, decide the family by vote.
3. **Features per (client, family)**: last charge, billing period, projected next charge, timing vs. the
   other families, stream quality, refunds; plus client-level features (refund ratio, activity trend, …).
4. **Two-stage LightGBM**: binary scorer per (client, family) → 8-class classifier on the 7 scores +
   client features. Two pipelines (v1: row-level evidence, v3: noise-robust streams), 3 seeds each,
   averaged 50/50.

## Data usage
- Training labels: **train only** (2,000 clients) + 5 noisy copies per client (noise injected into
  descriptions/MCCs to match the much noisier test histories; labels unchanged).
- Validation: **scoring only**.
- Test histories: prediction input; also used (unlabeled) to calibrate the noise level of the copies.
- Unlabeled pretrain set: analysis only (tolerances, timing), not used by the script.

## Results (macro-F1, validation set, never trained on)
| Model | Macro-F1 |
|---|---|
| v1 | 0.585 |
| v3 | 0.598 |
| **Blend (submitted)** | **0.616** |

## Key findings
- Label = family of the active subscription that bills first after the cutoff.
- Noise rises by design from pretrain → train → valid → test (in test only ~32% of subscription charges
  keep an informative description); training on noisy copies of train gave the largest gain.
- About a third of apparently active subscriptions stop at the cutoff, which caps achievable accuracy.
