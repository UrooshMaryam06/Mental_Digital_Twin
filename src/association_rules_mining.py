import pandas as pd
import numpy as np
import pickle
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def _safe_qcut(series, labels):
    try:
        return pd.qcut(series, q=3, labels=labels, duplicates="drop")
    except ValueError:
        ranked = series.rank(method="average")
        return pd.qcut(ranked, q=3, labels=labels, duplicates="drop")


def _time_period(hour):
    if hour < 6:
        return "time_NIGHT"
    if hour < 12:
        return "time_MORNING"
    if hour < 18:
        return "time_AFTERNOON"
    return "time_EVENING"


def _get_season(month):
    if month in [12, 1, 2]:
        return "season_WINTER"
    if month in [3, 4, 5]:
        return "season_SPRING"
    if month in [6, 7, 8]:
        return "season_SUMMER"
    return "season_AUTUMN"


def _engineer_features(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").set_index("time")

    df["renewable"] = (
        df["generation solar"].fillna(0)
        + df["generation wind onshore"].fillna(0)
        + df["generation hydro run-of-river and poundage"].fillna(0)
        + df["generation hydro water reservoir"].fillna(0)
        + df["generation biomass"].fillna(0)
    )
    df["fossil"] = (
        df["generation fossil gas"].fillna(0)
        + df["generation fossil hard coal"].fillna(0)
    )
    df["nuclear"] = df["generation nuclear"].fillna(0)
    df["total_gen"] = df["renewable"] + df["fossil"] + df["nuclear"]
    df["renewable_pct"] = np.where(df["total_gen"] > 0, (df["renewable"] / df["total_gen"]) * 100, 0)

    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    df = df.dropna(subset=["total load actual"]).copy()

    df["demand_level"] = _safe_qcut(df["total load actual"], ["demand_LOW", "demand_MED", "demand_HIGH"])
    df["price_level"] = _safe_qcut(df["price actual"], ["price_LOW", "price_MED", "price_HIGH"])
    df["renewable_level"] = _safe_qcut(df["renewable_pct"], ["renewable_LOW", "renewable_MED", "renewable_HIGH"])
    df["fossil_level"] = _safe_qcut(df["fossil"], ["fossil_LOW", "fossil_MED", "fossil_HIGH"])

    df["time_period"] = df["hour"].apply(_time_period)
    df["season"] = df["month"].apply(_get_season)
    df["weekend_flag"] = df["is_weekend"].map({1: "is_WEEKEND", 0: "is_WEEKDAY"})

    return df


def run_association_mining(data_path="energy_dataset.csv"):
    """
    Run Apriori association rule mining on energy dataset.
    Discovers patterns between demand levels, price levels,
    renewable/fossil generation, time of day, season, and weekend.

    Returns:
        rules (pd.DataFrame): Association rules with support, confidence, lift
    """
    df = pd.read_csv(data_path)
    df = _engineer_features(df)

    transaction_cols = [
        "demand_level",
        "price_level",
        "renewable_level",
        "fossil_level",
        "time_period",
        "season",
        "weekend_flag",
    ]

    df = df.dropna(subset=transaction_cols).copy()
    transactions = df[transaction_cols].astype(str).values.tolist()

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(
        df_encoded,
        min_support=0.05,
        use_colnames=True,
        max_len=3,
    )
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    print(f"Found {len(frequent_itemsets)} frequent itemsets")

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1.2,
    )

    rules = rules[
        (rules["confidence"] >= 0.5)
        & (rules["lift"] >= 1.2)
    ].copy()
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    print(f"Generated {len(rules)} association rules after filtering")

    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/association_rules.pkl", "wb") as f:
        pickle.dump(rules, f)

    rules_csv = rules.copy()
    rules_csv["antecedents"] = rules_csv["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules_csv["consequents"] = rules_csv["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules_csv = rules_csv[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules_csv.to_csv("artifacts/association_rules.csv", index=False)

    print("Artifacts saved: artifacts/association_rules.pkl and artifacts/association_rules.csv")

    print("\n" + "=" * 70)
    print("TOP 15 ASSOCIATION RULES BY LIFT")
    print("=" * 70)
    for i, row in rules.head(15).iterrows():
        antecedents = ", ".join(sorted(row["antecedents"]))
        consequents = ", ".join(sorted(row["consequents"]))
        print(f"  [{i + 1:2d}] {antecedents}")
        print(f"       → {consequents}")
        print(
            f"          support={row['support']:.3f}  confidence={row['confidence']:.3f}  lift={row['lift']:.3f}"
        )
        print()

    return rules


if __name__ == "__main__":
    rules = run_association_mining()
    print(f"\nDone. {len(rules)} rules saved to artifacts/")
