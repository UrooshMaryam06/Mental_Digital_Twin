import pandas as pd
import pytest
import os

RULES_PATH = 'artifacts/association_rules.csv'


@pytest.fixture
def rules():
    if not os.path.exists(RULES_PATH):
        pytest.skip("Association rules not generated yet. Run src/association_rules_mining.py first.")
    return pd.read_csv(RULES_PATH)


def test_rules_not_empty(rules):
    assert len(rules) > 0, "Should have at least one rule"


def test_required_columns(rules):
    for col in ['antecedents', 'consequents', 'support', 'confidence', 'lift']:
        assert col in rules.columns


def test_confidence_range(rules):
    assert (rules['confidence'] >= 0).all() and (rules['confidence'] <= 1).all()


def test_lift_positive(rules):
    assert (rules['lift'] > 1.0).all(), "All rules should have lift > 1.0"


def test_support_range(rules):
    assert (rules['support'] > 0).all() and (rules['support'] <= 1).all()
