Overall challenge #2 spot winners (DueNext).
<img width="908" height="726" alt="file-57462c597e4f480a058bed66c4477ebd" src="https://github.com/user-attachments/assets/b1d12b44-c8c3-49de-967e-522b8b427393" />

Read the challenge here: https://github.com/Swiss-ai-Weeks/ubs-2026

# Predicting the next subscription: the whole pipeline

## Internal validation scorecard

**Macro-F1: 0.6338** on the 1,000-client development validation set, reproduced
from the pipeline in this repository. This is an internal development result,
not a hidden-test score: the Fausto specialist blend was calibrated on this same
validation set. The Fausto model scored **0.6154 with its standalone class offsets**;
LightGBM alone scored **0.6196**. A fixed 50/50 average of their probability
tables scored **0.6338** without retraining the Fausto models.

| Category | Actual clients | Correct | Predicted as category | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| cloud | 89 | 65 | 94 | 0.691 | 0.730 | 0.710 |
| gym | 121 | 81 | 130 | 0.623 | 0.669 | 0.645 |
| insurance | 99 | 65 | 104 | 0.625 | 0.657 | 0.640 |
| mobile | 104 | 65 | 99 | 0.657 | 0.625 | 0.640 |
| music | 93 | 37 | 62 | 0.597 | 0.398 | 0.477 |
| software | 104 | 64 | 102 | 0.627 | 0.615 | 0.621 |
| streaming | 97 | 49 | 80 | 0.613 | 0.505 | 0.554 |
| none | 293 | 243 | 329 | 0.739 | 0.829 | 0.781 |

For each row, precision means `correct / predicted as category` and recall means
`correct / actual clients`. F1 combines the two; **macro-F1 averages the eight
category F1 scores equally**, regardless of how many clients belong to each one.
The headline 0.6338 uses the unrounded category scores; table values are rounded
to three decimals.

**Final submission CSV:** [`submission.csv`](submission.csv) is present at the
repository root and is produced by `python forecasting_optimized/ensemble_predict.py`. It
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

The model combines two views. The Fausto specialists study schedules, amounts,
candidate families, description embeddings, and merchant-like streams. A second
LightGBM pipeline trains on noisy copies of the training histories so it can still
recognize subscriptions when descriptions and category codes are unreliable. Each
view returns eight scores; we average the two tables and choose the largest score.

On the fixed 1,000-client validation set, it has a macro-F1 of **0.6338**. The
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
    -> train Fausto specialists on train; tune their blend on validation
    -> refit the two main structured models on train + validation
    -> separately train noise-robust LightGBM on train plus noisy train copies
    -> average the two probability tables 50/50; predict each test client
    -> write submission.csv in the template's client order
```

The exact commands are in [Run the complete pipeline](#run-the-complete-pipeline).
The committed ZIPs let you start at the raw data or skip ahead to ready-made
feature CSVs. Test labels are not part of the input.

## How the data is used

| Split | Clients | Transactions | How we use it |
|---|---:|---:|---|
| Train | 2,000 | 147,459 | Fit the supervised specialists, vocabulary, and numeric scaling; make five noisy copies for LightGBM training. |
| Validation | 1,000 | 73,898 | Choose Fausto model settings and calibration; score both model views and their fixed 50/50 average. |
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

The LightGBM view starts from the raw JSONL histories. It builds amount-and-timing
subscription streams in two ways, creates five noisy copies of each train client,
and fits two-stage models across three seeds. The first stage scores each
`(client, family)` pair; five folds keep copies of the same client together when
making training scores for the second stage. Validation labels are used for scoring
this LightGBM view, not for fitting it. The amount of injected text/MCC noise was
chosen after inspecting **unlabeled test histories**; test labels were not used.

The main client/family feature builder rejects transactions at or after the
2026-01-01 cutoff. The optional historical proxy stage uses later transactions **only** to
label earlier historical cutoffs in the unlabeled pretraining set. For the final
test predictions, the two main structured models and seven family models are
refit on train plus validation after their settings are chosen; the other
specialists use their saved training-stage fits. LightGBM uses train and noisy train
copies. The final CSV averages the two test probability tables; the development
score averages their validation tables. Because part of the Fausto view is refit
before test inference, the two evaluations do not use exactly the same fitted
models.

## Repository in one glance

| Path | Job |
|---|---|
| `data/*.zip` | Raw splits and prepared feature splits; extracted CSV/JSONL files are generated locally. |
| `dataset_cleaning/clean_dataset.py` | Turn raw transactions into the prepared feature archive. |
| `transaction_embedding/` | Embed original descriptions and prepare per-transaction arrays for the text specialist. |
| `forecasting_optimized/features.py` | Build client, family, and merchant-stream feature tables. |
| `forecasting_optimized/*_experiment.py` and `experiment.py` | Train the specialists and save validation probabilities. |
| `forecasting_optimized/blend_models.py` | Combine specialists and save the decision rule. |
| `forecasting_optimized/predict.py` | Apply saved Fausto models and export their test probabilities. |
| `forecasting_optimized/noise_robust_lightgbm.py` | Train and score the noise-robust LightGBM stream models. |
| `forecasting_optimized/ensemble_predict.py` | Average the two model views and write the final submission. |
| `submission.csv` | The ready-to-submit predictions, committed at the repository root. |

Each committed Python module either generates an input, trains a specialist, combines
their outputs, or runs final inference. Intermediate data and model artifacts stay
out of Git; the final CSV is committed.

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
scikit-learn 1.9.1, CatBoost 1.2.10, XGBoost 3.4.1, LightGBM 4.7.0,
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
python forecasting_optimized/noise_robust_lightgbm.py
```

The scripts write generated models, probability tables, and the blend configuration
under `forecasting_optimized/artifacts/`. Training is CPU-intensive; the embedding
generation stage can use an available accelerator. The validation labels guide
early stopping, model selection, blend calibration, and the reported score, so the
score is a **development-set result**. After settings are chosen, the main
structured models and seven family specialists are refit on `train + valid` and
saved as `model_final.joblib` and `family_models_final.joblib`. Submission inference
prefers those 3,000-client final artifacts; the other specialists use their saved
training-stage fits. The separate LightGBM script trains on the 2,000 labeled train
clients plus five noisy copies per client; it writes validation and test probability
tables under `forecasting_optimized/artifacts/lightgbm/` and fits no validation
labels.

### 5. Create the submission

```bash
python forecasting_optimized/predict.py \
  --output-csv forecasting_optimized/artifacts/fausto_only_submission.csv \
  --probabilities-npz forecasting_optimized/artifacts/fausto_test_probabilities.npz
python forecasting_optimized/ensemble_predict.py
```

The Fausto predictor exports its **pre-offset** probability table; the LightGBM
script has already exported the other table. `ensemble_predict.py` checks both
tables against validation labels, then averages aligned test probabilities 50/50.
Its output is [`submission.csv`](submission.csv) at the repository root, with the
same client order and schema as `data/dataset/sample_submission.csv`. The checked-in
file is already ready to submit; these commands are for reproduction.

Extracted data, embeddings, trained artifacts, and caches are ignored by Git. The
final submission CSV is tracked so it is available without rebuilding the models.

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

Each view produces eight scores, one for each allowed label. The final prediction is
the label with the largest **50/50 averaged** score. It does not forecast the exact
merchant, amount, or date.

The data flow is:

```text
transactions before the cutoff
          |
          +--> Fausto features + specialists ----------> eight scores --+
          |                                                            |
          +--> noisy training copies + LightGBM streams -> eight scores --+--> 50/50 average
                                                                         |
                                                                         v
                                                                  highest score: one label
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

### 8. Noise-robust LightGBM streams

This independent view tries to follow a subscription even when its next charge
arrives with a generic description or misleading MCC. One stream builder groups
likely family charges first; another groups charges by stable amount and billing
schedule, then attaches noisy charges that fit. LightGBM first scores each possible
`(client, family)` pair and then compares those seven scores with client-wide
features to choose among all eight labels. Five client-grouped folds create training
scores for the second stage; three random seeds are averaged. The two stream
builders are averaged 50/50 to make one LightGBM probability table.

## Step 4: combine the votes

The Fausto view first combines its own specialists. Its initial table is:

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

The **final** submission combines the resulting Fausto table with the independent
LightGBM table using a fixed equal average:

```text
final_scores = 0.5 * Fausto_scores + 0.5 * LightGBM_scores
prediction   = label with the largest final score
```

The Fausto table here is taken *before* its own log-score class offsets. Its other
specialist blends and class-specific adjustments still apply. Both models produce
scores in the same eight-label order and are aligned by `client_id` before
averaging. On this development split, Fausto alone scores 0.6154 **with its own
offsets**, LightGBM alone scores 0.6196, and the equal average of the pre-offset
Fausto table and LightGBM table scores 0.6338 macro-F1.

## Step 5: measure macro-F1

The training and blending scripts tune macro-F1. For each class:

```text
precision = correct predictions of this class / all predictions of this class
recall    = correct predictions of this class / all real examples of this class
F1        = 2 * precision * recall / (precision + recall)
macro-F1  = average of the eight class F1 values
```

Every class gets one eighth of macro-F1, even though `none` has 293 validation
examples and `cloud` has only 89. Fausto's standalone model experiments used
class offsets to balance their decisions:

```text
adjusted_score[class] = log(probability[class]) + offset[class]
prediction            = class with the largest adjusted_score
```

Those offsets raised the Fausto-only development score, but **the final 50/50
ensemble does not use them**. It takes the class-adjusted Fausto probability table,
averages it with LightGBM, and takes the largest result.

The 50/50 blend was inspected on the same validation set used to report its result.
The score therefore remains a development result, even though the LightGBM models
did not fit validation labels.

## What is actually proven by the result?

“Proof” here means reproducible empirical evidence, not a mathematical theorem.

### The measured evidence

The [scorecard](#internal-validation-scorecard) shows precision, recall, and F1 for
every category. Their equally weighted average produces the internal macro-F1.

The totals are:

- reproduced final ensemble macro-F1: **0.6338**;
- Fausto view alone with its offsets: **0.6154**; noise-robust LightGBM alone:
  **0.6196**;
- earlier reported attention model: approximately **0.440 macro-F1**;
- difference from that earlier report on this split: about **0.194** (roughly
  **44%** relative). The earlier run was not repeated in this environment.

The comparison suggests three reasons for the improvement on this split:

1. Explicit recurrence features are very valuable for this dataset.
2. Treating each family as a comparable candidate helps the model share examples
   across categories.
3. Diverse specialist models plus class-aware decisions improve the fixed validation
   score.

### What the result does not prove

It does not prove that unseen-test macro-F1 is 0.6338. The validation set influenced:

- early stopping in several base models;
- candidate model and hyperparameter selection;
- ensemble weights;
- sequential inclusion of specialists;
- eight class offsets;
- class-specific adjustments;
- inspection of the final 50/50 blend.

The structured feature builder rejects transactions on or after the cutoff. The
reported metric still has **selection bias** because the validation labels
helped construct the final decision rule. These are two different meanings of
“leakage,” and passing the first does not solve the second.

The honest description is therefore:

> The ensemble achieved a macro-F1 of 0.6338 on the development
> validation set. Its unbiased generalization score still needs confirmation.

## Where the model still fails

The validation errors show useful patterns:

- `music` is hardest: 37 of 93 are correct. It is often confused with insurance,
  mobile, streaming, and other digital subscriptions.
- `streaming` has 49 of 97 correct and is often confused with gym, cloud, music, and
  `none`.
- `none` recall is 0.829: 243 of 293 are found, but 86 other clients are also
  predicted as `none`.
- `software` has precision 0.627 and recall 0.615, so it remains imperfect in
  both directions.
- `gym` recall is 0.669; 40 of 121 real gym cases are missed.

Those are more actionable than the single 0.6338 number.

## Future improvements, in priority order

1. **Measure generalization.** Use an untouched test set or nested cross-validation
   to assess how much of the development score carries over to new clients.
2. **Simplify and calibrate the blend.** Generate out-of-fold predictions for every
   specialist, test which models add value, and learn a regularized combination.
3. **Improve subscription tracking.** Combine text, amount, and timing to distinguish
   merchants, handle price changes, and estimate when each payment will happen next.
4. **Focus on music and streaming.** Review their errors and test whether better
   merchant grouping or description features can separate these weaker categories.
5. **Strengthen final training and checks.** Validate historical proxy labels, refit
   all selected components consistently on allowed labeled data, and check cutoff
   handling, client coverage, class order, and reproducibility.

Evaluate one change at a time and keep it only when the gain survives evaluation
on data that was not used to select that change.
