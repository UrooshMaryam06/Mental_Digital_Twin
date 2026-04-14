import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("/content/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")

print("Original shape:", df.shape)

# 2. Clean column names (VERY IMPORTANT)
# -------------------------
df.columns = df.columns.str.strip()

# 3. Keep only useful columns
# -------------------------
keep_cols = [
    "user_id",
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9",
    "happiness.score",
    "phq.day",
    "sex",
    "age",
    "time",
    "period.name"
]

df = df[keep_cols]
# numeric columns
num_cols = [
    "phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9",
    "happiness.score",
    "phq.day",
    "age"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())


def depression_label(phq9):
    if phq9 <= 4:
        return "low"
    elif phq9 <= 9:
        return "mild"
    elif phq9 <= 14:
        return "moderate"
    else:
        return "severe"
    
df["phq9_total"] = df[["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]].sum(axis=1)
df["depression_level"] = df["phq9_total"].apply(depression_label)

df = pd.get_dummies(df, columns=["depression_level"])

# categorical columns
df["sex"] = df["sex"].fillna("unknown")
df["period.name"] = df["period.name"].fillna("unknown")


# 4. Encode sex properly
# -------------------------
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["sex"] = df["sex"].fillna(0).astype(int)


# 5. One-hot encoding for period.name
# -------------------------
df = pd.get_dummies(df, columns=["period.name"], drop_first=False)

# 6. Feature engineering from time
# -------------------------
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df = df.drop(columns=["time"])


# fill any missing time-derived values
df["hour"] = df["hour"].fillna(df["hour"].median())
df["day"] = df["day"].fillna(df["day"].median())
df["month"] = df["month"].fillna(df["month"].median())


# convert all boolean columns to int
bool_cols = df.select_dtypes(include=["bool"]).columns
df[bool_cols] = df[bool_cols].astype(int)

# 7. Final check
# -------------------------
print("Final shape:", df.shape)
print(df.head())
