# Vehicle Insurance Claims: Does a Decision Model Help XGBoost Catch Fraud?

XGBoost is the standard model for fraud scoring. We tested whether it gets better at spotting
fraudulent auto insurance claims when it also gets a decision model's answers to a few plain-language
questions about each claim. We compared **Grayson** (Finic) and **Jev** (TypeSafe).

Write-up: [finic.ai/blog/decision-models-xgboost-insurance-fraud](https://finic.ai/blog/decision-models-xgboost-insurance-fraud)

## Results

| Approach | ROC-AUC | vs XGBoost | Repeats better than XGBoost |
| --- | --- | --- | --- |
| XGBoost | 0.798 | — | — |
| XGBoost + Jev | 0.796 | −0.003 | 4 of 12 |
| **XGBoost + Grayson** | **0.805** | **+0.007** | **10 of 12** |

ROC-AUC is 0.5 for random guessing and 1.0 for a perfect ranking. Each number is the average of 12
repeats of 5-fold cross-validation over all 1,782 claims.

What that means for an insurer with 100,000 claims a year that sends 10% of them to investigators,
assuming a $5,000 average claim and half of each caught fraud's value saved:

| Approach | Extra frauds caught per year | Savings per year |
| --- | --- | --- |
| XGBoost + Jev | −25 | −$64k |
| **XGBoost + Grayson** | **+165** | **+$412k** |

## Why Grayson helps

Grayson turns a claim's raw codes into the concepts an adjuster actually thinks in: does this policy
pay for the policyholder's own car, were they at fault, did they move just before the claim? Those
were the Grayson answers XGBoost relied on most. With only about 1,400 claims to learn from in each
run, XGBoost struggles to assemble those concepts from raw categories on its own; Grayson hands them
over ready-made. On real claims, where notes and statements carry information no column captures, we
expect Grayson to add new signal as well, not just better-shaped signal.

## How it works

1. **Data.** Oracle's public [CLAIMS](https://github.com/AnalyticsandDataOracleUserCommunity/MachineLearning/tree/master/claims)
   extract: 1,782 auto claims from a US insurer (1994–96), 923 of them fraud, with 27 fields each.
2. **Questions.** Eight yes/no questions, such as "Was the policyholder at fault?" and "Did the
   address change shortly before the claim?", chosen from a training subset of the claims.
3. **Answers.** Every claim goes to Grayson and Jev once with the same questions. Neither model sees
   whether the claim was fraud.
4. **Models.** The same XGBoost model three times: on the claim fields alone, plus Jev's answers, and
   plus Grayson's answers.
5. **Comparison.** Repeated cross-validation, so every claim is scored by a model that never saw it.

The extract is 52% fraud, which doesn't affect ROC-AUC. For the savings estimate we use a real-world
rate of about 6%: policy numbers run from 1 to 15,420, and the extract holds 923 fraud claims.

## Run it yourself

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp
.venv/bin/jupyter nbconvert --to notebook --execute --inplace benchmark.ipynb
```

The cached API answers are in `data/api_responses/`, so no API keys are needed. To query the APIs
yourself, copy `.env.example` to `.env` and add `GRAYSON_API_KEY` and `TYPESAFE_API_KEY`.

## Files

```
benchmark.ipynb      the whole benchmark
fraudbench/          data loading, API clients and question selection
data/questions.json  the 8 questions
data/api_responses/  cached Grayson and Jev answers
results/             results tables and ROC curve
```
