"""Question list derivation, using the training split's labels only.

Each candidate question is paired with a predicate over the named fields. A question is
kept only if its predicate's fraud lift on the training split confirms it. The direct
question is always kept. The result is frozen to data/questions.json before any API call.
"""

import json
from pathlib import Path

import pandas as pd

QUESTIONS_PATH = Path("data/questions.json")
DIRECT = ("is_fraud", "Is this record fraudulent?")


def lift_table(train, candidates, label, min_lift, min_support):
    """candidates: [(name, question text, predicate fn, field(s) it reads)]."""
    overall = train[label].mean()
    rows = []
    for name, text, pred, fields in candidates:
        mask = pred(train).fillna(False).astype(bool)
        support = mask.mean()
        rate = train.loc[mask, label].mean() if mask.any() else float("nan")
        rows.append({"name": name, "question": text, "fields": fields, "support": support,
                     "fraud_rate": rate, "lift": rate / overall})
    t = pd.DataFrame(rows).set_index("name")
    t["kept"] = (t["lift"] >= min_lift) & (t["support"] >= min_support)
    t.attrs["rule"] = f"kept if training lift >= {min_lift} and support >= {min_support:.1%}"
    return t.sort_values("lift", ascending=False)


def freeze(table, path=QUESTIONS_PATH):
    kept = table[table["kept"]]
    questions = {n: q for n, q in kept["question"].items()}
    questions[DIRECT[0]] = DIRECT[1]
    Path(path).write_text(json.dumps({
        "questions": questions,
        "derivation": table.reset_index().round(4).to_dict(orient="records"),
        "rule": table.attrs.get("rule"),
    }, indent=2))
    return questions


def load(path=QUESTIONS_PATH):
    return json.loads(Path(path).read_text())["questions"]
