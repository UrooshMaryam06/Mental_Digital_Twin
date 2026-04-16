import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
# -----------------------
df = pd.read_csv("Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")
df.columns = df.columns.str.strip()

# 2. Select useful columns
# -----------------------
cols = [
    "user_id",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9",
    "happiness.score",
    "age",
    "sex",
    "time",
    "period.name"
]

df = df[cols]


# 3. Handle missing values
phq_cols = [f"phq{i}" for i in range(1,10)]

for col in phq_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df["happiness.score"] = pd.to_numeric(df["happiness.score"], errors="coerce")
df["happiness.score"] = df["happiness.score"].fillna(df["happiness.score"].median())

df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["age"] = df["age"].fillna(df["age"].median())


# 4. Encode sex
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# 5. Time feature engineering
df["time"] = pd.to_datetime(df["time"], errors="coerce")

df["hour"] = df["time"].dt.hour
df["weekday"] = df["time"].dt.dayofweek

df["hour"] = df["hour"].fillna(df["hour"].median())
df["weekday"] = df["weekday"].fillna(df["weekday"].median())

df.drop(columns=["time"], inplace=True)

# 1. create phq total
df["phq9_total"] = df[phq_cols].sum(axis=1)

# 2. create target
def label(x):
    if x <= 4:
        return 0
    elif x <= 9:
        return 1
    elif x <= 14:
        return 2
    else:
        return 3

df["target"] = df["phq9_total"].apply(label)

# 3. NOW remove it (important step)
df.drop(columns=["phq9_total"], inplace=True)

# clean missing values first
df = pd.get_dummies(df, columns=["period.name"], drop_first=False, dtype=int)

df["prev_target"] = df.groupby("user_id")["target"].shift(1)
df["prev_happiness"] = df.groupby("user_id")["happiness.score"].shift(1)

df["rolling_mean_mood"] = df.groupby("user_id")["target"].rolling(3).mean().reset_index(0, drop=True)

df["prev_target"] = df["prev_target"].fillna(df["target"])
df["prev_happiness"] = df["prev_happiness"].fillna(df["happiness.score"])
df["rolling_mean_mood"] = df["rolling_mean_mood"].fillna(df["happiness.score"])
# 7. Final dataset
# -----------------------
X = df.drop(columns=["target", "user_id"])
y = df["target"]
 df