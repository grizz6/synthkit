"""The synthkit command-line interface: fit, emit, check, diff, inspect, compare.

Kept as a thin layer over the library: every command here is a few lines of glue around
Profile, privacy.check, drift.compute_drift, or inspect.summarize_profile. Actual logic
belongs in those modules.
"""

from __future__ import annotations

from pathlib import Path

import typer

from synthkit import __version__, privacy
from synthkit.constraints import parse_constraints
from synthkit.drift import DEFAULT_DRIFT_THRESHOLD, compute_drift
from synthkit.inspect import compare_profiles, summarize_profile
from synthkit.io import read_table, write_table
from synthkit.privacy import DEFAULT_RARE_COMBINATION_THRESHOLD
from synthkit.profile import Profile
from synthkit.types import ColumnType

app = typer.Typer(add_completion=False, help="Synthetic test fixtures from a real dataset's shape.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"synthkit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the synthkit version and exit.",
    ),
) -> None:
    pass


def _parse_column_types(overrides: list[str]) -> dict[str, ColumnType]:
    """Parse repeated `--column-type name=type` flags into Profile.fit's column_types dict."""
    parsed: dict[str, ColumnType] = {}

    for override in overrides:
        name, separator, raw_type = override.partition("=")
        if not separator or not name:
            raise typer.BadParameter(
                f"expected NAME=TYPE, got {override!r} (e.g. --column-type signup_at=datetime)"
            )
        try:
            parsed[name] = ColumnType(raw_type)
        except ValueError:
            valid = ", ".join(sorted(t.value for t in ColumnType))
            raise typer.BadParameter(
                f"unknown column type {raw_type!r} for {name!r}; expected one of: {valid}"
            ) from None

    return parsed


@app.command()
def fit(
    data: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Real dataset to fit a profile on (.parquet or .csv).",
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Where to write profile.json."),
    constraints: Path | None = typer.Option(
        None,
        "--constraints",
        exists=True,
        dir_okay=False,
        readable=True,
        help="YAML file declaring business-rule constraints.",
    ),
    column_type: list[str] = typer.Option(
        [],
        "--column-type",
        help="Override inferred type as NAME=TYPE (e.g. signup_at=datetime). Repeatable.",
    ),
    holdout: float = typer.Option(
        0.0,
        "--holdout",
        help="Fraction of rows to hold out for an immediate self-check of DCR ratio (0 skips it).",
    ),
) -> None:
    """Fit a profile on real data. No real rows are written to `output`."""
    df = read_table(data)
    parsed_constraints = parse_constraints(constraints) if constraints else None
    column_types = _parse_column_types(column_type)

    unknown = sorted(set(column_types) - set(df.columns.astype(str)))
    if unknown:
        raise typer.BadParameter(f"--column-type names not present in {data}: {unknown}")

    profile = Profile.fit(df, column_types=column_types or None, constraints=parsed_constraints)
    profile.save(output)
    typer.echo(f"wrote profile for {len(df)} rows, {len(df.columns)} columns -> {output}")

    if holdout > 0:
        # The profile is already saved; a check() failure here (e.g. too few rows to split
        # into a holdout) should be a warning, not a crash of an otherwise successful fit.
        try:
            synthetic = profile.emit(n=len(df), seed=0)
            report = privacy.check(synthetic, df, profile.column_types, holdout_fraction=holdout)
            typer.echo(
                f"self-check: dcr_ratio={report.dcr_ratio:.3f} exact_matches={report.exact_matches}"
            )
        except ValueError as e:
            typer.echo(f"self-check skipped: {e}", err=True)


@app.command()
def emit(
    profile: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="profile.json produced by `synthkit fit`.",
    ),
    n: int = typer.Option(..., "-n", help="Number of synthetic rows to generate."),
    seed: int = typer.Option(..., "--seed", help="Random seed; same seed always emits same rows."),
    output: Path = typer.Option(..., "-o", "--output", help="Where to write the synthetic table."),
) -> None:
    """Emit synthetic rows from a committed profile. Never touches real data."""
    profile_obj = Profile.load(profile)
    synthetic = profile_obj.emit(n=n, seed=seed)
    write_table(synthetic, output)
    typer.echo(f"wrote {n} synthetic rows -> {output}")


@app.command()
def check(
    fixtures: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Synthetic table produced by `synthkit emit`.",
    ),
    profile: Path = typer.Option(
        ...,
        "--profile",
        exists=True,
        dir_okay=False,
        readable=True,
        help="profile.json the fixtures came from.",
    ),
    real: Path = typer.Option(
        ...,
        "--real",
        exists=True,
        dir_okay=False,
        readable=True,
        help="The original real dataset, for the DCR baseline.",
    ),
    min_dcr_ratio: float = typer.Option(
        1.0, "--min-dcr-ratio", help="Minimum acceptable synthetic/holdout DCR ratio."
    ),
    holdout_fraction: float = typer.Option(0.2, "--holdout-fraction"),
    rare_combination_threshold: int = typer.Option(
        DEFAULT_RARE_COMBINATION_THRESHOLD, "--rare-combination-threshold"
    ),
) -> None:
    """Verify synthetic fixtures against a privacy baseline. Exits non-zero on failure."""
    synthetic = read_table(fixtures)
    real_df = read_table(real)
    profile_obj = Profile.load(profile)

    try:
        report = privacy.check(
            synthetic,
            real_df,
            profile_obj.column_types,
            holdout_fraction=holdout_fraction,
            min_dcr_ratio=min_dcr_ratio,
            rare_combination_threshold=rare_combination_threshold,
        )
    except ValueError as e:
        typer.echo(f"check failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"dcr_ratio: {report.dcr_ratio:.3f} (min {min_dcr_ratio})")
    typer.echo(f"exact_matches: {report.exact_matches}")
    typer.echo(f"rare_combination_leaks: {report.rare_combination_leaks}")

    if not report.passed:
        typer.echo("FAILED privacy check", err=True)
        raise typer.Exit(code=1)

    typer.echo("PASSED privacy check")


@app.command()
def diff(
    profile: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="profile.json to compare against.",
    ),
    data: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Fresh data to check for drift away from the profile.",
    ),
    threshold: float = typer.Option(DEFAULT_DRIFT_THRESHOLD, "--threshold"),
) -> None:
    """Check whether fresh data has drifted away from the profile fixtures are built on."""
    profile_obj = Profile.load(profile)
    df = read_table(data)

    report = compute_drift(profile_obj, df, threshold=threshold)

    for column, score in sorted(report.column_drift.items(), key=lambda kv: -kv[1]):
        flag = " DRIFTED" if column in report.drifted_columns else ""
        typer.echo(f"{column}: {score:.3f}{flag}")

    if not report.passed:
        count = len(report.drifted_columns)
        typer.echo(f"FAILED: {count} column(s) drifted past {threshold}", err=True)
        raise typer.Exit(code=1)

    typer.echo("PASSED: no drift detected")


@app.command()
def inspect(
    profile: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="profile.json to summarize.",
    ),
) -> None:
    """Print a human-readable summary of a profile: columns, types, and shape."""
    profile_obj = Profile.load(profile)
    summaries = summarize_profile(profile_obj)

    typer.echo(
        f"{len(profile_obj.columns)} columns, {profile_obj.n_rows_fit} rows fit on, "
        f"{len(profile_obj.constraints)} constraint(s)"
    )
    typer.echo()

    name_width = max((len(s.name) for s in summaries), default=0)
    type_width = max((len(s.type) for s in summaries), default=0)

    for summary in summaries:
        copula_flag = "[copula]" if summary.in_copula else ""
        name_col = f"{summary.name:<{name_width}}"
        type_col = f"{summary.type:<{type_width}}"
        line = f"{name_col}  {type_col}  {copula_flag:<8}  {summary.detail}"
        if summary.null_rate is not None:
            line += f"  (null_rate={summary.null_rate:.3f})"
        typer.echo(line)


@app.command()
def compare(
    old: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="The profile as it was (e.g. from git).",
    ),
    new: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="The profile as it is now.",
    ),
) -> None:
    """Show what changed between two profiles, semantically rather than as a JSON diff."""
    comparison = compare_profiles(Profile.load(old), Profile.load(new))

    old_rows, new_rows = comparison.rows_fit
    old_constraints, new_constraints = comparison.constraint_counts
    typer.echo(f"rows fit on: {old_rows} -> {new_rows}")
    typer.echo(f"constraints: {old_constraints} -> {new_constraints}")
    typer.echo()

    if comparison.added:
        typer.echo(f"added:   {', '.join(comparison.added)}")
    if comparison.removed:
        typer.echo(f"removed: {', '.join(comparison.removed)}")

    for column in comparison.changed:
        typer.echo(f"changed: {column.name}")
        for change in column.changes:
            typer.echo(f"    {change}")

    if not comparison.any_changes:
        typer.echo("no column-level changes")
    else:
        typer.echo(f"\nunchanged: {len(comparison.unchanged)} column(s)")


if __name__ == "__main__":
    app()
