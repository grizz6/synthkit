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


def test_strict_inequality_flags_ties_as_violations():
    # Regression test: a strict "<" only checked for ">" (not ">="), so a tie (equal values)
    # was never flagged and passed through unrepaired, leaving the strict constraint violated.
    df = pd.DataFrame({"start": [5], "end": [5]})
    fixed = apply_constraints(df, [Inequality("start", "<", "end")])
    assert (fixed["start"] < fixed["end"]).all()


def test_strict_inequality_nudges_integer_ties_apart():
    df = pd.DataFrame({"start": [5, 1], "end": [5, 10]})
    fixed = apply_constraints(df, [Inequality("start", "<", "end")])
    assert (fixed["start"] < fixed["end"]).all()
    assert fixed["start"].iloc[1] == 1 and fixed["end"].iloc[1] == 10  # untouched, not tied


def test_strict_greater_than_nudges_ties_in_the_correct_direction():
    df = pd.DataFrame({"a": [5], "b": [5]})
    fixed = apply_constraints(df, [Inequality("a", ">", "b")])
    assert (fixed["a"] > fixed["b"]).all()


def test_non_strict_inequality_still_allows_ties():
    df = pd.DataFrame({"a": [5], "b": [5]})
    fixed = apply_constraints(df, [Inequality("a", "<=", "b")])
    assert fixed["a"].iloc[0] == 5 and fixed["b"].iloc[0] == 5


def test_strict_inequality_nudges_float_ties_by_the_smallest_representable_step():
    df = pd.DataFrame({"start": [5.0], "end": [5.0]})
    fixed = apply_constraints(df, [Inequality("start", "<", "end")])
    assert fixed["start"].iloc[0] < fixed["end"].iloc[0]
    # A "smallest step" nudge should be imperceptibly small, not a large jump.
    assert fixed["end"].iloc[0] - fixed["start"].iloc[0] < 1e-9


def test_strict_inequality_nudges_datetime_ties_by_one_second():
    df = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01 00:00:00"]),
            "end": pd.to_datetime(["2024-01-01 00:00:00"]),
        }
    )
    fixed = apply_constraints(df, [Inequality("start", "<", "end")])
    assert fixed["start"].iloc[0] < fixed["end"].iloc[0]
    assert (fixed["end"].iloc[0] - fixed["start"].iloc[0]) == pd.Timedelta(seconds=1)


def test_strict_inequality_leaves_string_ties_unresolved():
    # Documented limitation: there's no well-defined "smallest step" for strings, so a tie
    # between two string columns is left as-is rather than guessed at.
    df = pd.DataFrame({"a": ["m"], "b": ["m"]})
    fixed = apply_constraints(df, [Inequality("a", "<", "b")])
    assert fixed["a"].iloc[0] == "m" and fixed["b"].iloc[0] == "m"


def test_multi_column_unique_preserves_non_colliding_rows():
    # Regression test: multi-column Unique used to overwrite every listed column for every
    # row with a fabricated token, discarding realistic sampled values wholesale instead of
    # only disambiguating rows whose combination actually collides.
    df = pd.DataFrame(
        {
            "first_name": ["Alice", "Bob", "Carol"],
            "last_name": ["Smith", "Jones", "Smith"],
        }
    )
    fixed = apply_constraints(
        df, [Unique(["first_name", "last_name"])], rng=np.random.default_rng(0)
    )
    assert fixed["first_name"].tolist() == ["Alice", "Bob", "Carol"]
    combos = list(zip(fixed["first_name"], fixed["last_name"], strict=True))
    assert len(set(combos)) == len(combos)


def test_multi_column_unique_disambiguates_actual_collisions():
    df = pd.DataFrame(
        {
            "first_name": ["Alice", "Alice", "Alice"],
            "last_name": ["Smith", "Smith", "Smith"],
        }
    )
    fixed = apply_constraints(
        df, [Unique(["first_name", "last_name"])], rng=np.random.default_rng(0)
    )
    combos = list(zip(fixed["first_name"], fixed["last_name"], strict=True))
    assert len(set(combos)) == 3
    assert (fixed["first_name"] == "Alice").all()  # only the last column was disambiguated


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
    fixed = apply_constraints(df, [ConditionalNull("cancelled_at", "status != 'cancelled'")])
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
