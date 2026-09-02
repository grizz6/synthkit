"""A human-readable summary of a fitted profile.

A committed profile.json is meant to be reviewed like any other artifact in a repo, but its
raw JSON (rounded floats, quantile knots, frequency tables) isn't something a reviewer can
read at a glance. This turns a Profile back into a short per-column summary: type, shape, null
rate, and whether it's tied to other columns through the copula.
"""

from __future__ import annotations

from dataclasses import dataclass

from synthkit.marginals import (
    BooleanMarginal,
    CategoricalMarginal,
    DatetimeMarginal,
    IdentifierMarginal,
    NumericMarginal,
    TextMarginal,
)
from synthkit.profile import ALL_NULL_KIND, MARGINAL_CLASS_BY_KIND, Profile


@dataclass
class ColumnSummary:
    name: str
    type: str
    detail: str
    null_rate: float | None
    in_copula: bool


def _null_rate(profile: Profile, column: str) -> float | None:
    columns = profile.null_model.get("columns", [])
    if column not in columns:
        return None

    index = columns.index(column)
    patterns = profile.null_model["patterns"]
    probabilities = profile.null_model["probabilities"]
    return sum(
        prob for pattern, prob in zip(patterns, probabilities, strict=True) if pattern[index]
    )


def _describe(marginal_dict: dict) -> str:
    kind = marginal_dict["kind"]

    if kind == ALL_NULL_KIND:
        return "always null"

    marginal = MARGINAL_CLASS_BY_KIND[kind].from_dict(marginal_dict)

    if isinstance(marginal, NumericMarginal):
        low, high = marginal.quantile_values[0], marginal.quantile_values[-1]
        return f"range [{low:.3g}, {high:.3g}]"

    if isinstance(marginal, CategoricalMarginal):
        detail = f"{len(marginal.categories)} categories"
        if marginal.other_mass > 0:
            detail += f", other_mass={marginal.other_mass:.3f}"
        return detail

    if isinstance(marginal, BooleanMarginal):
        return f"P(true)={marginal.probability_true:.3f}"

    if isinstance(marginal, DatetimeMarginal):
        low, high = marginal.numeric.quantile_values[0], marginal.numeric.quantile_values[-1]
        granularity = marginal.granularity_seconds
        return f"range [{int(low)}s, {int(high)}s] (epoch), granularity={granularity}s"

    if isinstance(marginal, IdentifierMarginal):
        return f"{marginal.style} (regenerated, never modeled statistically)"

    if isinstance(marginal, TextMarginal):
        return f"free text, {len(marginal.word_pool)}-word vocabulary"

    return "unknown"


def summarize_profile(profile: Profile) -> list[ColumnSummary]:
    """One ColumnSummary per column, in the profile's original column order."""
    return [
        ColumnSummary(
            name=column,
            type=profile.column_types[column],
            detail=_describe(profile.marginals[column]),
            null_rate=_null_rate(profile, column),
            in_copula=column in profile.copula_columns,
        )
        for column in profile.columns
    ]


@dataclass
class ColumnChange:
    name: str
    changes: list[str]


@dataclass
class ProfileComparison:
    added: list[str]
    removed: list[str]
    changed: list[ColumnChange]
    unchanged: list[str]
    rows_fit: tuple[int, int]
    constraint_counts: tuple[int, int]

    @property
    def any_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)


def _format_null_rate(rate: float | None) -> str:
    return "none" if rate is None else f"{rate:.3f}"


def _column_changes(old: ColumnSummary, new: ColumnSummary) -> list[str]:
    changes = []
    if old.type != new.type:
        changes.append(f"type {old.type} -> {new.type}")
    if old.detail != new.detail:
        changes.append(f"{old.detail} -> {new.detail}")
    # Compared at display precision: re-fitting shifts these slightly every time, and a
    # difference too small to print isn't a difference worth reporting.
    if _format_null_rate(old.null_rate) != _format_null_rate(new.null_rate):
        changes.append(
            f"null_rate {_format_null_rate(old.null_rate)} -> {_format_null_rate(new.null_rate)}"
        )
    if old.in_copula != new.in_copula:
        changes.append("joined the copula" if new.in_copula else "left the copula")
    return changes


def compare_profiles(old: Profile, new: Profile) -> ProfileComparison:
    """What actually changed between two profiles, semantically rather than textually.

    A re-fitted profile's raw JSON diff is a wall of shifted quantile knots, which tells a
    pull-request reviewer nothing. This reports the things a reviewer actually cares about:
    columns that appeared or disappeared, types that changed, and shape or null rates that
    moved.
    """
    old_by_name = {s.name: s for s in summarize_profile(old)}
    new_by_name = {s.name: s for s in summarize_profile(new)}

    added = [name for name in new_by_name if name not in old_by_name]
    removed = [name for name in old_by_name if name not in new_by_name]

    changed: list[ColumnChange] = []
    unchanged: list[str] = []
    for name, new_summary in new_by_name.items():
        if name not in old_by_name:
            continue
        changes = _column_changes(old_by_name[name], new_summary)
        if changes:
            changed.append(ColumnChange(name=name, changes=changes))
        else:
            unchanged.append(name)

    return ProfileComparison(
        added=added,
        removed=removed,
        changed=changed,
        unchanged=unchanged,
        rows_fit=(old.n_rows_fit, new.n_rows_fit),
        constraint_counts=(len(old.constraints), len(new.constraints)),
    )
