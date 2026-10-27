"""Loading, encoding, the question-selection split, candidate questions and API contexts for Oracle CLAIMS."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
SOURCE_URL = ("https://raw.githubusercontent.com/AnalyticsandDataOracleUserCommunity/"
              "MachineLearning/master/claims/CLAIMS.csv")
LABEL = "fraud"

# The extract is rebalanced: all 923 fraud claims of a 15,420-claim table (policy numbers run
# 1-15,420) plus 859 legitimate ones. The full table's fraud rate is therefore 923 / 15,420.
TRUE_RATE = 923 / 15420

RENAME = {
    "FRAUDFOUND": "fraud", "WEEKOFMONTH": "week_of_month", "DAYOFWEEK": "day_of_week",
    "MAKE": "make", "ACCIDENTAREA": "accident_area", "DAYOFWEEKCLAIMED": "day_of_week_claimed",
    "MONTHCLAIMED": "month_claimed", "WEEKOFMONTHCLAIMED": "week_of_month_claimed", "SEX": "sex",
    "MARITALSTATUS": "marital_status", "FAULT": "fault", "VEHICLECATEGORY": "vehicle_category",
    "VEHICLEPRICE": "vehicle_price", "POLICYNUMBER": "policy_number", "REPNUMBER": "rep_number",
    "DEDUCTIBLE": "deductible", "DRIVERRATING": "driver_rating",
    "DAYS:POLICY-ACCIDENT": "days_policy_accident", "DAYS:POLICY-CLAIM": "days_policy_claim",
    "PASTNUMBEROFCLAIMS": "past_number_of_claims", "AGEOFVEHICLE": "age_of_vehicle",
    "AGEOFPOLICYHOLDER": "age_of_policyholder", "POLICEREPORTFILED": "police_report_filed",
    "WITNESSPRESENT": "witness_present", "AGENTTYPE": "agent_type",
    "NUMBEROFSUPPLIMENTS": "number_of_supplements", "ADDRESSCHANGE-CLAIM": "address_change_claim",
    "NUMBEROFCARS": "number_of_cars", "BASEPOLICY": "base_policy",
}

# Ordered bins, lowest first, ordinal-encoded for XGBoost. Every other text column is nominal.
ORDINAL = {
    "vehicle_price": ["less than 20_000", "20_000 to 29_000", "30_000 to 39_000",
                      "40_000 to 59_000", "60_000 to 69_000", "more than 69_000"],
    "days_policy_accident": ["none", "1to7", "8to15", "15to30", "morethan30"],
    "days_policy_claim": ["none", "1to7", "8to15", "15to30", "morethan30"],
    "past_number_of_claims": ["none", "1", "2to4", "morethan4"],
    "age_of_vehicle": ["new", "2years", "3years", "4years", "5years", "6years", "7years", "morethan7"],
    "age_of_policyholder": ["16to17", "18to20", "21to25", "26to30", "31to35", "36to40",
                            "41to50", "51to65", "over65"],
    "number_of_supplements": ["none", "1to2", "3to5", "morethan5"],
    "address_change_claim": ["nochange", "under6months", "1year", "2to3years", "4to8years"],
    "number_of_cars": ["1vehicle", "2vehicles", "3to4", "5to8", "morethan8"],
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "day_of_week_claimed": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "month_claimed": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
}


def download_raw(raw_dir=RAW_DIR):
    path = Path(raw_dir) / "CLAIMS.csv"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)
    return path


def load(raw_dir=RAW_DIR):
    """CLAIMS.csv with normalized names, a 0/1 label and a record id (the policy number)."""
    df = pd.read_csv(download_raw(raw_dir)).rename(columns=RENAME)
    df[LABEL] = (df[LABEL] == "Yes").astype(int)
    # The policy number is an identifier: it becomes the record id and is never a feature.
    return df.rename(columns={"policy_number": "id"})


def xgb_features(df):
    """Ordinal codes for ordered bins, label codes for nominal columns."""
    X = df.drop(columns=["id", LABEL, "split"], errors="ignore").copy()
    for c in X.columns:
        if c in ORDINAL:
            X[c] = X[c].map({v: i for i, v in enumerate(ORDINAL[c])}).astype("float32")
        elif not pd.api.types.is_numeric_dtype(X[c]):
            codes, _ = pd.factorize(X[c], sort=True)
            X[c] = np.where(codes < 0, np.nan, codes).astype("float32")
    return X


def assign_split(df, seed):
    """Stratified 60/20/20. There is no year column, so a time split is impossible."""
    train, rest = train_test_split(df["id"], test_size=0.4, stratify=df[LABEL], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=df.set_index("id").loc[rest, LABEL],
                                 random_state=seed)
    df["split"] = np.select([df["id"].isin(train), df["id"].isin(val)], ["train", "val"], "test")
    return df


def candidates(train):
    """[(name, question, predicate, fields)]. The first group are common insurance red flags; the
    rest were added from training-set lift. Every one must pass the lift rule to be kept."""
    o = {c: {v: i for i, v in enumerate(ORDINAL[c])} for c in ORDINAL}
    veh_age = lambda d: d["age_of_vehicle"].map(o["age_of_vehicle"])
    price = lambda d: d["vehicle_price"].map(o["vehicle_price"])
    return [
        ("at_fault", "Was the policyholder at fault?",
         lambda d: d["fault"] == "PolicyHolder", "fault"),
        ("no_police_report", "Was no police report filed?",
         lambda d: d["police_report_filed"] == "No", "police_report_filed"),
        ("no_witness", "Was there no witness?",
         lambda d: d["witness_present"] == "No", "witness_present"),
        ("address_changed", "Did the address change shortly before the claim?",
         lambda d: d["address_change_claim"] != "nochange", "address_change_claim"),
        ("claim_soon_after_policy", "Was the claim filed soon after the policy started?",
         lambda d: d["days_policy_claim"] != "morethan30", "days_policy_claim"),
        ("prior_claims", "Does the policyholder have prior claims?",
         lambda d: d["past_number_of_claims"] != "none", "past_number_of_claims"),
        ("old_for_price", "Is the vehicle old relative to its price band?",
         lambda d: (veh_age(d) >= o["age_of_vehicle"]["7years"])
         & (price(d) >= o["vehicle_price"]["40_000 to 59_000"]), "age_of_vehicle, vehicle_price"),
        # Added from training lift.
        ("first_claim", "Is this the policyholder's first claim?",
         lambda d: d["past_number_of_claims"] == "none", "past_number_of_claims"),
        ("covers_own_vehicle",
         "Does the policy cover damage to the policyholder's own vehicle, not just liability?",
         lambda d: d["base_policy"] != "Liability", "base_policy"),
        ("rural_accident", "Did the accident happen in a rural area?",
         lambda d: d["accident_area"] == "Rural", "accident_area"),
        ("young_policyholder", "Is the policyholder 25 or younger?",
         lambda d: d["age_of_policyholder"].map(o["age_of_policyholder"]) <= o["age_of_policyholder"]["21to25"],
         "age_of_policyholder"),
        ("sport_car", "Is the vehicle a sports car?",
         lambda d: d["vehicle_category"] == "Sport", "vehicle_category"),
    ]


def build_context(row):
    """Every field except the label and the policy number, with bins kept as text."""
    return {
        "record_type": "auto insurance claim (US insurer, 1994-96)",
        "accident": {
            "day_of_week": row["day_of_week"], "week_of_month": int(row["week_of_month"]),
            "area": row["accident_area"],
        },
        "claim": {
            "day_of_week_claimed": row["day_of_week_claimed"], "month_claimed": row["month_claimed"],
            "week_of_month_claimed": int(row["week_of_month_claimed"]),
            "fault": row["fault"], "police_report_filed": row["police_report_filed"],
            "witness_present": row["witness_present"],
            "number_of_supplements": row["number_of_supplements"],
            "claims_rep_number": int(row["rep_number"]),
        },
        "vehicle": {
            "make": row["make"], "category": row["vehicle_category"],
            "price_usd": row["vehicle_price"], "age": row["age_of_vehicle"],
        },
        "policyholder": {
            "sex": row["sex"], "marital_status": row["marital_status"],
            "age": row["age_of_policyholder"], "driver_rating_1_to_4": int(row["driver_rating"]),
            "past_number_of_claims": row["past_number_of_claims"],
            "address_change_before_claim": row["address_change_claim"],
            "number_of_cars": row["number_of_cars"],
        },
        "policy": {
            "base_policy": row["base_policy"], "deductible_usd": int(row["deductible"]),
            "agent_type": row["agent_type"],
            "days_from_policy_start_to_accident": row["days_policy_accident"],
            "days_from_policy_start_to_claim": row["days_policy_claim"],
        },
    }
