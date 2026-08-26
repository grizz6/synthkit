import numpy as np
import pandas as pd

from synthkit.constraints import ConditionalNull, Derived, ForeignKey, Inequality, Unique
from synthkit.repair import apply_constraints


def test_inequality_swap_fixes_violations():
    df = pd.DataFrame({"created_at": [5, 1, 3], "updated_at": [2, 4, 3]})
    fixed = apply_constraints(df, [Inequality("created_at", "<=", "updated_at")])
    assert (fixed["created_at"] <= fixed["updated_at"]).all()


def test_inequality_swap_preserves_marginal_values():
    # Swapping should keep the same multiset of values in each column, not clamp them.
    df = pd.DataFrame({"a": [5.0, 1.0, 3.0], "b": [2.0, 4.0, 3.0]})
    fixed = apply_constraints(df, [Inequality("a", "<=", "b")])
    combined_before = sorted(df["a"].tolist() + df["b"].tolist())
    combined_after = sorted(fixed["a"].tolist() + fixed["b"].tolist())
    assert combined_before == combined_after


def test_inequality_leaves_satisfied_rows_untouched():
    df = pd.DataFrame({"a": [1, 2], "b": [5, 6]})
    fixed = apply_constraints(df, [Inequality("a", "<=", "b")])
    pd.testing.assert_frame_equal(fixed, df)


def test_derived_column_is_recomputed():
    df = pd.DataFrame({"subtotal": [10.0, 20.0], "tax": [1.0, 2.0], "total": [0.0, 0.0]})
    fixed = apply_constraints(df, [Derived("total", "subtotal + tax")])
    assert fixed["total"].tolist() == [11.0, 22.0]


def test_conditional_null_applies_rule():
    df = pd.DataFrame(
        {
            "status": ["active", "cancelled", "active"],
            "cancelled_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    fixed = apply_constraints(
        df, [ConditionalNull("cancelled_at", "status != 'cancelled'")]
    )
    assert fixed["cancelled_at"].isna().tolist() == [True, False, True]
    assert fixed["cancelled_at"].iloc[1] == "2024-01-02"


def test_unique_single_column_produces_distinct_values():
    df = pd.DataFrame({"customer_id": [1, 1, 1, 1, 1]})
    fixed = apply_constraints(df, [Unique(["customer_id"])], rng=np.random.default_rng(0))
    assert fixed["customer_id"].nunique() == len(fixed)


def test_foreign_key_samples_from_provided_pool():
    df = pd.DataFrame({"customer_id": [0, 0, 0, 0]})
    fixed = apply_constraints(
        df,
        [ForeignKey("customer_id", "customers.id")],
        rng=np.random.default_rng(0),
        key_pools={"customer_id": [101, 102, 103]},
    )
    assert set(fixed["customer_id"]).issubset({101, 102, 103})


def test_foreign_key_without_pool_leaves_column_unchanged():
    df = pd.DataFrame({"customer_id": [7, 7, 7]})
    fixed = apply_constraints(df, [ForeignKey("customer_id", "customers.id")])
    assert fixed["customer_id"].tolist() == [7, 7, 7]


def test_constraints_applied_in_correct_order():
    # Derived must run before inequality repair sees the derived column, and conditional
    # null must run after both.
    df = pd.DataFrame(
        {
            "subtotal": [10.0],
            "tax": [1.0],
            "total": [0.0],
            "cap": [5.0],
        }
    )
    fixed = apply_constraints(
        df,
        [
            Derived("total", "subtotal + tax"),
            Inequality("cap", "<=", "total"),
        ],
    )
    assert fixed["total"].iloc[0] == 11.0
    assert fixed["cap"].iloc[0] <= fixed["total"].iloc[0]
