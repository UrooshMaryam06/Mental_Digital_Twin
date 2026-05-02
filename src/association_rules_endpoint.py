import pickle
import pandas as pd
from typing import Optional


def load_rules(path='artifacts/association_rules.pkl') -> pd.DataFrame:
    """Load saved association rules from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def _interpret_rule(row) -> str:
    """Generate a human-readable interpretation of a rule."""
    ant = ', '.join(sorted(row['antecedents']))
    con = ', '.join(sorted(row['consequents']))
    conf_pct = round(row['confidence'] * 100, 1)
    return (
        f"When [{ant}], then [{con}] "
        f"with {conf_pct}% confidence (lift={row['lift']:.2f})"
    )


def query_rules(
    rules: pd.DataFrame,
    demand_level: Optional[str] = None,
    price_level: Optional[str] = None,
    renewable_level: Optional[str] = None,
    top_n: int = 5
) -> list[dict]:
    """
    Given partial energy state, find matching association rules.
    Returns top_n rules sorted by lift.
    """
    filters = []
    if demand_level:
        filters.append(f'demand_{demand_level}')
    if price_level:
        filters.append(f'price_{price_level}')
    if renewable_level:
        filters.append(f'renewable_{renewable_level}')

    if not filters:
        matched = rules.head(top_n)
    else:
        mask = rules['antecedents'].apply(
            lambda ant: all(f in ant for f in filters)
        )
        matched = rules[mask].head(top_n)

    results = []
    for _, row in matched.iterrows():
        results.append({
            'antecedents': sorted(list(row['antecedents'])),
            'consequents': sorted(list(row['consequents'])),
            'support': round(float(row['support']), 4),
            'confidence': round(float(row['confidence']), 4),
            'lift': round(float(row['lift']), 4),
            'interpretation': _interpret_rule(row)
        })

    return results
