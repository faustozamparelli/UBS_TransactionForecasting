# Predicting the next subscription: the whole pipeline

## Internal validation scorecard

**Macro-F1: 0.6154** on the 1,000-client development validation set, reproduced
from the pipeline in this repository. This is an internal development result,
not a hidden-test score: model selection and calibration used the same validation
set. The final offset adjustment improved this score from 0.6090 to 0.6154 without
retraining the base models.

| Category | Actual clients | Correct | Predicted as category | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| cloud | 89 | 59 | 97 | 0.608 | 0.663 | 0.634 |
| gym | 121 | 97 | 154 | 0.630 | 0.802 | 0.705 |
| insurance | 99 | 66 | 113 | 0.584 | 0.667 | 0.623 |
| mobile | 104 | 74 | 125 | 0.592 | 0.712 | 0.646 |
| music | 93 | 47 | 107 | 0.439 | 0.505 | 0.470 |
| software | 104 | 57 | 84 | 0.679 | 0.548 | 0.606 |
| streaming | 97 | 48 | 86 | 0.558 | 0.495 | 0.525 |
| none | 293 | 188 | 234 | 0.803 | 0.642 | 0.713 |

For each row, precision means `correct / predicted as category` and recall means
`correct / actual clients`. F1 combines the two; **macro-F1 averages the eight
category F1 scores equally**, regardless of how many clients belong to each one.
The headline 0.6154 uses the unrounded category scores; table values are rounded
to three decimals.

**Final submission CSV:** [`submission.csv`](submission.csv) is present at the
repository root and is produced by `python forecasting_optimized/predict.py`. It
contains one row per test client, in the sample template's order, with exactly two columns:
`client_id` and `predicted_next_recurring_merchant`. The second column holds one of
the seven merchant families or `none`. It is committed so judges can use it without
retraining the ensemble. The repository does **not** commit the much larger model
artifacts, so rerunning `predict.py` from a fresh clone requires completing the
training steps below first.

Think of the model as a detective looking at a client's bank statement. It cannot
see the future. It can spot that a gym charge keeps arriving around the third of
each month, distinguish that pattern from a one-off grocery trip, and guess which
recurring category will show up first. This guide starts with the raw files and
ends with the submission CSV. No machine-learning background is needed.

## The 30-second explanation

The model looks at everything a client paid **before 2026-01-01** and answers one
question:

> Which recurring merchant family is most likely to produce the first recurring
> payment during the next 90 days?

It must choose exactly one answer:

`cloud`, `gym`, `insurance`, `mobile`, `music`, `software`, `streaming`, or `none`.

The model is a committee of smaller models. Some study schedules and amounts, one
compares each possible family separately, one summarizes description embeddings,
and others study merchant-like streams. Their scores are blended and then adjusted
so that rare classes are not automatically ignored.

On the fixed 1,000-client validation set, it has a macro-F1 of **0.6154**. The
earlier attention model was reported at about **0.440** macro-F1 on the same split;
that earlier run was not reproduced here. The result suggests that this design fits
the development task better. It does not establish the score on new clients, because the
validation labels were also used to choose and calibrate many parts of the ensemble.

## Route map

```text
ZIP of raw JSONL transactions + labeled client answers
    -> extract the train / validation / test splits
    -> clean descriptions, suggest candidate families, add past-only features
    -> summarize each client's history and each client/family pair
    -> embed original descriptions for a second view of the history
    -> train specialists on train; compare and tune on validation
    -> refit the two main structured models on train + validation
    -> blend saved scores, adjust for macro-F1, predict each test client
    -> write submission.csv in the template's client order
```

The exact commands are in [Run the complete pipeline](#run-the-complete-pipeline).
The committed ZIPs let you start at the raw data or skip ahead to ready-made
feature CSVs. Test labels are not part of the input.

## How the data is used

| Split | Clients | Transactions | How we use it |
|---|---:|---:|---|
| Train | 2,000 | 147,459 | Fit the supervised specialists, vocabulary, and numeric scaling. |
| Validation | 1,000 | 73,898 | Choose model settings, blend weights, class adjustments, and final offsets; measure the internal score. |
| Test | 1,000 | 75,761 | Predict one label per client for `submission.csv`; no test labels are available. |
| Historical pretraining | Unlabeled client histories | — | Make approximate labels at earlier cutoffs to train the proxy specialists. |

The raw transactions and labels live in `data/dataset.zip`; the matching prepared
feature CSVs live in `data/dataset_features.zip`. Both archives are committed. The
cleaning script can rebuild the feature ZIP from raw JSONL. It keeps every
transaction, sorts by client and time, cleans description text, assigns a rough
candidate family, and adds calendar and past-history fields. The structured
forecasters rebuild client-level summaries from those prepared rows. The text
specialist separately embeds the *original* descriptions, fitting numeric scales
and category dictionaries on train only, then applying them to validation and test.

The main client/family feature builder rejects transactions at or after the
2026-01-01 cutoff. The optional historical proxy stage uses later transactions **only** to
label earlier historical cutoffs in the unlabeled pretraining set. For the final
test predictions, the two main structured models and seven family models are
refit on train plus validation after their settings are chosen; the other
specialists use their saved training-stage fits. This is why the development
validation score and the exact models used for test inference are related but not
identical evaluations.

## Repository in one glance

| Path | Job |
|---|---|
| `data/*.zip` | Raw splits and prepared feature splits; extracted CSV/JSONL files are generated locally. |
| `dataset_cleaning/clean_dataset.py` | Turn raw transactions into the prepared feature archive. |
| `transaction_embedding/` | Embed original descriptions and prepare per-transaction arrays for the text specialist. |
| `forecasting_optimized/features.py` | Build client, family, and merchant-stream feature tables. |
| `forecasting_optimized/*_experiment.py` and `experiment.py` | Train the specialists and save validation probabilities. |
| `forecasting_optimized/blend_models.py` | Combine specialists and save the decision rule. |
| `forecasting_optimized/predict.py` | Apply saved models to test clients and write the submission. |
| `submission.csv` | The ready-to-submit predictions, committed at the repository root. |

Each committed Python module either generates an input, trains a specialist, combines
their outputs, or runs final inference. Intermediate data and model artifacts stay
out of Git; the final CSV is committed.

## First: what is a recurring payment?

Imagine these payments in one client's history:

| Date | Description | Amount | Candidate family |
|---|---|---:|---|
| 3 October | Fit Club Online | 39.99 | gym |
| 3 November | Billing Fit Club | 39.99 | gym |
| 3 December | Fit Club Core | 39.99 | gym |
| 12 December | Video Stream | 18.99 | streaming |
| 18 December | Grocery Store | 64.20 | none |

The three gym payments are a strong recurring pattern:

- the description is similar;
- the amount is stable;
- the gap is about one month;
- the last payment was recent;
- another payment is expected inside the next 90 days.

The grocery payment is recent, but it is not a recurring-merchant candidate. The
streaming payment belongs to a relevant family, but one observation is weak evidence
of recurrence. A reasonable model should therefore favor `gym`.

That small example is essentially the entire problem. The real implementation repeats
this reasoning for seven possible families, thousands of clients, noisy descriptions,
missing payments, irregular schedules, and clients with several merchants in the same
family.

## The input and output, literally

For each client, the input is a chronological list of transactions. A transaction can
contain fields such as:

- timestamp;
- amount and currency;
- direction and type;
- merchant category code (MCC);
- raw description;
- cleaned description;
- history-only helper values.

The output is eight scores, one for each allowed label. The final prediction is the
label with the largest **adjusted** score. It does not forecast the exact merchant,
amount, or date.

The data flow is:

```text
transactions before the cutoff
          |
          +--> simple MCC/text rules --> candidate family per transaction
          |
          +--> counts, recency, cadence, amount, and stream features
          |                            |
          |                            +--> several tree models
          |
          +--> raw-description embeddings --> pooled-embedding model
                                               |
all model probability tables -----------------+
          |
          v
weighted blend --> per-class adjustments --> class offsets --> one label
```

## Step 1: give each transaction a rough family

The cleaning code first applies understandable rules using the MCC, description, and
sometimes amount. For example, a gym MCC or words such as `gym`, `fitness`, or
`fit club` make the transaction a `gym` candidate. Obvious groceries, salaries, ATM
withdrawals, and similar noise become `none`.

This rule is a **candidate generator**, not the final prediction. It says “this
transaction might be gym,” not “the client's next recurring payment will be gym.”
The distinction matters: the forecasting models still have to decide whether the
candidate actually repeats and whether it is more convincing than the other families.

Every transaction is retained. Non-candidate transactions still contribute to global
client activity and pooled embedding features. `none` here means "no candidate
family for this transaction"; the final client-level `none` prediction means "no
recurring family is forecast in the next 90 days." These are different questions.

## Step 2: turn a messy history into useful numbers

Machine-learning models are much better at a question such as “the median gap is 30
days and the last payment was 29 days ago” than at rediscovering that fact from a long
transaction table with limited labeled data. `features.py` therefore builds two main
views.

### Client-wide view

There is one row per client. It contains general client activity plus a block of
features for every family. This lets a multiclass model compare all families at once.

Examples include:

- total transactions and recent transactions;
- number and share of candidate transactions;
- counts over the last 30, 60, 90, 180, and 365 days;
- time since the first and latest observation;
- mean, median, minimum, maximum, and recent gaps;
- gap variation and median absolute deviation;
- average amount and amount variation;
- number of descriptions and dominant-description share;
- calendar stability, such as variation in day of month.

### Client/family view

There are seven rows per client: one for each recurring family. Each row asks a binary
question such as “does this client have a convincing gym recurrence?” A shared model
can learn that a stable 30-day schedule matters whether the family is gym, mobile, or
software.

This creates seven times as many training rows and lets families share statistical
strength. It also includes the family's rank within the client: which family is most
frequent and which was seen most recently.

### The cadence calculation in plain English

Suppose a client paid on 3 October, 3 November, and 3 December:

```text
gaps                    = [31 days, 30 days]
median gap              = 30.5 days
days from last payment
to the cutoff           = 29 days
cycles since last       = 29 / 30.5 = 0.95
projected next payment  = about 1.5 days after the cutoff
inside 90-day horizon   = yes
```

That is a strong recurrence signal. By contrast, gaps of 8, 73, and 19 days with
wildly changing amounts are much weaker even if the total count is the same.

### Merchant streams prevent averages from lying

A client may have two software subscriptions: one costs 9.99 monthly and another
costs 79.00 quarterly. If all software transactions are mixed, their gaps and amounts
look chaotic. The code clusters transactions into approximate amount-stable streams
using a 12% relative tolerance, then computes recurrence features per stream.

Another specialist groups normalized descriptions. Decorations such as `billing`,
`member`, `online`, and `core` are removed, so `Billing Fit Club` and `Fit Club
Online` can become the same merchant-like stream.

These methods are useful approximations, but neither is true merchant identity. That
is an important improvement opportunity discussed later.

## Step 3: ask several models instead of trusting one

The ensemble uses different models because each representation has different blind
spots.

### 1. CatBoost multiclass model

This model sees the entire client-wide feature row and directly chooses among all
eight labels. It is good at interactions such as “mobile is recent, but gym is both
recent and much more regular.”

### 2. Shared pair model

This model sees one `(client, family)` row at a time. It learns a general definition
of recurrence across all families. It is the largest member of the initial blend,
with weight 0.601.

### 3. Family-specific models

Seven binary CatBoost models are trained, one per family. They can learn that regular
insurance may behave differently from weekly or monthly entertainment payments.

### 4. Out-of-fold stack

The stack learns from pair-model and CatBoost predictions. Its training predictions
are out-of-fold: each training client is predicted by base models that did not train
on that client. This prevents the stack from learning from unrealistically perfect
in-sample predictions.

### 5. Historical proxy models

Unlabeled histories are turned into approximate supervised examples at three earlier
cutoffs: 2025-08-01, 2025-09-01, and 2025-10-01. A future payment is treated as a
continuation only when the past stream has enough observations, plausible cadence,
stable amount, and acceptable phase error.

These labels are noisy, so real labeled examples receive four times their weight.

### 6. XGBoost pair model and family ranker

The extra pair model provides a different tree-learning bias. The ranker separates
two questions:

1. Will there be no recurring payment?
2. If there will be one, which of the seven families should rank first?

This matches the logical structure of the task better than forcing one model to solve
both questions at once.

### 7. Pooled embeddings and description streams

MiniLM turns each *original* description into 384 numbers representing its text.
The preprocessor also records category codes, scaled numeric values, calendar cycles,
and missing-value flags. It learns its category dictionary and numeric scales on
train, then reuses them unchanged on validation and test. The pooled-embedding
specialist summarizes those values over the full history and most recent 12
transactions; an XGBoost model learns from the summaries. The description-stream
specialist separately scores recurrence after grouping similar cleaned
descriptions. These views help when hand-written family features miss information
in the text.

## Step 4: combine the votes

The initial probability table is:

```text
P = 0.314 * CatBoost
  + 0.601 * shared pair model
  + 0.085 * family-specific models
```

Those weights sum to one. Extra sources are then added one at a time using:

```text
P_new = (1 - w) * P_old + w * P_specialist
```

The saved sequential weights are 0.272 for stacking, 0.181 for proxy CatBoost, 0.151
for the proxy pair model, 0.296 for the XGBoost pair model, 0.203 for ranking, 0.479
for pooled embeddings, and 0.161 for description streams.

These numbers must **not** be read as final global percentages because every later
step shrinks the mixture produced by earlier steps. Some classes then receive small
specialist-specific adjustments as well.

Why blend at all? Imagine two models:

- Model A recognizes clean monthly schedules but confuses music and streaming.
- Model B understands descriptions but is less reliable about cadence.

If their errors are not identical, averaging them can preserve both strengths. A
model does not need the best standalone score to add useful information.

## Step 5: adjust the decision for macro-F1

The training and blending scripts tune macro-F1. For each class:

```text
precision = correct predictions of this class / all predictions of this class
recall    = correct predictions of this class / all real examples of this class
F1        = 2 * precision * recall / (precision + recall)
macro-F1  = average of the eight class F1 values
```

Every class gets one eighth of macro-F1, even though `none` has 293 validation
examples and `cloud` has only 89. A raw highest-probability decision tends to favor
common classes. The code therefore adds a learned offset to each class's log score:

```text
adjusted_score[class] = log(probability[class]) + offset[class]
prediction            = class with the largest adjusted_score
```

The current offsets boost most merchant classes by different amounts and reduce
`none` by 0.4. This deliberately trades some `none` recall for more predictions of
smaller classes, which can improve macro-F1.

This is not cheating by itself; choosing a decision rule for the real metric is
normal. The risk is choosing many offsets and adjustments on the same validation set
used to report the final score.

## What is actually proven by the result?

“Proof” here means reproducible empirical evidence, not a mathematical theorem.

### The measured evidence

The [scorecard](#internal-validation-scorecard) shows precision, recall, and F1 for
every category. Their equally weighted average produces the internal macro-F1.

The totals are:

- reproduced macro-F1: **0.6154**;
- earlier reported attention model: approximately **0.440 macro-F1**;
- difference on this development split: about **0.175** (roughly **40%** relative).

The comparison suggests three reasons for the improvement on this split:

1. Explicit recurrence features are very valuable for this dataset.
2. Treating each family as a comparable candidate helps the model share examples
   across categories.
3. Diverse specialist models plus class-aware decisions improve the fixed validation
   score.

### What the result does not prove

It does not prove that unseen-test macro-F1 is 0.6154. The validation set influenced:

- early stopping in several base models;
- candidate model and hyperparameter selection;
- ensemble weights;
- sequential inclusion of specialists;
- eight class offsets;
- class-specific adjustments.

The structured feature builder rejects transactions on or after the cutoff. The
reported metric still has **selection bias** because the validation labels
helped construct the final decision rule. These are two different meanings of
“leakage,” and passing the first does not solve the second.

The honest description is therefore:

> The ensemble achieved a calibrated macro-F1 of 0.6154 on the development
> validation set. Its unbiased generalization score still needs confirmation.

## Where the model still fails

The validation errors show useful patterns:

- `music` is hardest: 47 of 93 are correct. It is often confused with insurance,
  mobile, streaming, and other digital subscriptions.
- `streaming` has 48 of 97 correct and is often confused with gym, cloud, music, and
  `none`.
- the model over-predicts merchant classes relative to `none`: it finds only 188 of
  293 real `none` cases. This is partly a deliberate consequence of macro-F1
  calibration.
- `software` has good precision (0.679) but weaker recall (0.548), meaning its
  predictions are fairly trustworthy but many real software cases are missed.
- `gym` has high recall (0.802) but lower precision (0.630), meaning it catches most
  gyms while also calling too many other clients gym.

Those are more actionable than the single 0.6154 number.

## How to improve it, in priority order

### 1. Obtain an honest score before adding more models

This is the highest-priority improvement because it changes how every later experiment
is judged.

Use nested or repeated cross-validation:

```text
outer fold:  completely untouched; used only for the final score
inner folds: train models, choose hyperparameters, blend, and tune offsets
```

Alternatively, freeze the current pipeline exactly and submit once to a truly hidden
test/leaderboard. Do not tune again from that result. Report the mean, standard
deviation, and per-class F1 across outer folds rather than one lucky split.

Expected benefit: not necessarily a higher displayed score, but a trustworthy one.

### 2. Cross-fit every source and learn a simpler blend

The current stack uses proper out-of-fold predictions, but most other specialists are
blended directly on the development validation set. Produce out-of-fold probabilities
for **every** base model, then train one regularized meta-model on those probabilities.

Replace dozens of greedy decisions with a constrained blend:

- non-negative weights;
- weights that sum to one;
- strong regularization;
- a small number of cross-fitted class thresholds;
- bootstrap checks that reject unstable weights.

Expected benefit: less calibration overfit, easier reasoning, and a smaller gap
between development and hidden-test performance.

### 3. Model merchant identity instead of approximating it with amount bands

The 12% amount clusters can merge unrelated merchants with similar prices and split a
real merchant after a price change. Description normalization can also merge distinct
products.

Build a merchant-entity layer using:

- normalized description tokens;
- MCC and currency;
- text-embedding similarity;
- amount and timing compatibility;
- explicit handling of price increases;
- client-local clustering, because merchant text may be synthetic or ambiguous.

Then calculate cadence per inferred merchant, and map the merchant to a family only
afterward. Evaluate entity clusters separately with sampled manual inspection or
synthetic ground truth.

Expected benefit: cleaner schedules, especially when one client has multiple
subscriptions in one family.

### 4. Predict the event process, not only the final class

The task is about which recurring event happens first inside 90 days. A more natural
model estimates a time-to-next-event distribution for each merchant stream:

```text
P(stream produces an event on day d | history)
```

The seven family probabilities can then be derived from competing risks: every stream
competes to be the first future event, while `none` is the probability that no stream
fires inside the horizon.

Useful families of methods include discrete-time survival models, renewal-process
features, gradient-boosted hazard models, or a neural temporal point process if enough
data exists.

Expected benefit: principled handling of missed cycles, annual subscriptions, and the
`none` decision.

### 5. Create more real-looking training cutoffs

The proxy-label idea is good, but its rule-generated labels inherit the assumptions of
the feature engineering. Generate rolling historical examples wherever enough future
history exists, and measure proxy-label precision on labeled or manually verified
subsets.

Improvements include:

- several cutoffs per client with group-aware folds so one client never enters both
  train and validation;
- confidence weights based on proxy-label quality;
- soft labels rather than forced labels when continuation is ambiguous;
- cutoff sampling that matches the real horizon and seasonality.

Expected benefit: substantially more training examples without pretending all proxy
labels are equally correct.

### 6. Attack music and streaming as a focused subproblem

Music and streaming descriptions and prices overlap. After the model decides that a
digital subscription is likely, use a specialist that distinguishes these two classes
with raw text, MCC, merchant entity, amount band, and history.

Before training it, review a stratified sample from these four groups:

- real music predicted as streaming;
- real streaming predicted as music;
- real music predicted as another family;
- real streaming predicted as another family.

If the raw fields do not contain enough information, no architecture can reliably
recover it; the right improvement would then be better data or a less granular label.

Expected benefit: direct gains on the two weakest per-class F1 values.

### 7. Calibrate probabilities before optimizing the final decision

Class offsets optimize decisions but do not guarantee that a score of 0.7 means a 70%
chance. On cross-fitted predictions, measure log loss, Brier score, and reliability
curves. Try temperature scaling, vector scaling, or carefully regularized one-vs-rest
calibration.

Then tune the macro-F1 decision rule only inside inner folds. Compare it with a simpler
cost-sensitive rule and verify that the improvement survives every outer fold.

Expected benefit: more stable blending and scores that other systems can safely use.

### 8. Refit the entire selected pipeline on all allowed labeled data

The predictor currently refits the two main structured models and seven family
specialists on train plus validation; its other specialists retain their
training-stage fits. Once the architecture and calibration are frozen, a complete
final-training stage could retrain every selected base model and the cross-fitted
meta-model using all allowed labeled data, without touching hidden labels.

Expected benefit: better use of scarce labels and consistent training provenance.

### 9. Add robustness and ablation tests

For every major source, remove it and measure the cross-validated score change. Also
test invariants:

- a post-cutoff transaction must raise an error in every feature path;
- shuffling input rows must not change a client's result;
- every prediction file has every client exactly once and only allowed labels;
- missing candidate families and all-noise histories remain valid;
- train/test feature columns and probability class order are identical;
- saved-model inference reproduces stored validation probabilities;
- fixed seeds reproduce scores within an explicit tolerance.

Expected benefit: fewer silent pipeline bugs and proof that complexity earns its cost.

## A practical experiment sequence

Do not implement all improvements at once. A clean sequence is:

1. Freeze the current pipeline and record an untouched or nested-CV baseline.
2. Build out-of-fold predictions for every current model.
3. Replace greedy blending and hand adjustments with one regularized meta-model.
4. Run ablations and remove sources that add no stable outer-fold gain.
5. Improve merchant identity and rerun the exact same evaluation.
6. Add a competing-risk/time-to-event model and compare it fairly.
7. Perform focused music/streaming error analysis.
8. Freeze, refit all selected components, and make the final test prediction.

At each step, keep a change only if its gain appears across folds and improves either
the target macro-F1 or a clearly stated secondary property such as calibration,
latency, or simplicity.

## Final verdict

The model is good because it encodes the structure of the problem: recurrence is
mostly about merchant identity, cadence, recency, amount stability, and competition
between possible families. The original sequence model had to discover all of that
from limited labels. The optimized ensemble gives those facts directly to models and
uses text and merchant-stream specialists for what the summaries miss.

The strongest next move is not another specialist. It is a stricter evaluation and a
simpler cross-fitted ensemble. Once the real generalization score is known, merchant
identity and explicit time-to-event modeling are the most promising technical
improvements.

## Run the complete pipeline

The repository intentionally keeps only the optimized forecasting pipeline and its
direct dependencies. Run every command below from the repository root.

### 1. Create the environment

Python 3.10 or newer is required. The embedding step downloads
`sentence-transformers/all-MiniLM-L6-v2` the first time it runs, so that first run
requires internet access.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -e transaction_embedding
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

The checked-in CSV was generated with Python 3.12.13, NumPy 2.5.3, pandas 3.0.6,
scikit-learn 1.9.1, CatBoost 1.2.10, XGBoost 3.4.1,
sentence-transformers 6.1.0, and PyTorch 2.14.0. `requirements.txt` gives compatible
version ranges, so a fresh install may select different versions and yield a
slightly different validation score or CSV.

### 2. Extract the input data

The committed ZIP files are the source of truth. Their extracted directories are
generated locally and ignored by Git.

```bash
mkdir -p data/dataset data/dataset_features
unzip -q data/dataset.zip -d data/dataset
unzip -q data/dataset_features.zip -d data/dataset_features
```

The optimized pipeline reads the ready-made feature archive. To rebuild that archive
from the raw JSONL transactions instead, run this optional command after extracting
`data/dataset.zip`:

```bash
python dataset_cleaning/clean_dataset.py
```

### 3. Generate the transaction embeddings

This embeds the original transaction descriptions with MiniLM and writes the train,
validation, and test arrays under
`transaction_embedding/artifacts/uncleaned/`:

```bash
python transaction_embedding/generate_uncleaned_embeddings.py
```

To select a device explicitly, add `--device cpu`, `--device cuda`, or
`--device mps`.

### 4. Train every ensemble component

Run the scripts in this order because later models load artifacts produced by earlier
ones:

```bash
python forecasting_optimized/experiment.py
python forecasting_optimized/family_experiment.py
python forecasting_optimized/stacking_experiment.py
python forecasting_optimized/proxy_pretrain_experiment.py
python forecasting_optimized/xgboost_experiment.py
python forecasting_optimized/ranking_experiment.py
python forecasting_optimized/embedding_pool_experiment.py
python forecasting_optimized/description_stream_experiment.py
python forecasting_optimized/blend_models.py
```

The scripts write generated models, probability tables, and the blend configuration
under `forecasting_optimized/artifacts/`. Training is CPU-intensive; the embedding
generation stage can use an available accelerator. The validation labels guide
early stopping, model selection, blend calibration, and the reported score, so the
score is a **development-set result**. After settings are chosen, the main
structured models and seven family specialists are refit on `train + valid` and
saved as `model_final.joblib` and `family_models_final.joblib`. Submission inference
prefers those 3,000-client final artifacts; the other specialists use their saved
training-stage fits.

### 5. Create the submission

```bash
python forecasting_optimized/predict.py
```

The output is [`submission.csv`](submission.csv) at the repository root, with the
same client order and schema as `data/dataset/sample_submission.csv`. The checked-in
file is already ready to submit; these commands are for reproduction.

Extracted data, embeddings, trained artifacts, and caches are ignored by Git. The
final submission CSV is tracked so it is available without rebuilding the models.
