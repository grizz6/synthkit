import json
import re

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from synthkit import __version__
from synthkit.cli import app
from synthkit.profile import Profile

runner = CliRunner()


def test_version_flag_prints_version_and_exits_zero():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def make_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.normal(100, 20, n),
            "plan_tier": rng.choice(["basic", "pro"], size=n, p=[0.7, 0.3]),
        }
    )


def test_fit_writes_profile(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"

    result = runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    assert result.exit_code == 0, result.output
    assert profile_path.exists()


def test_fit_with_holdout_prints_self_check(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"

    result = runner.invoke(
        app, ["fit", str(data_path), "-o", str(profile_path), "--holdout", "0.2"]
    )

    assert result.exit_code == 0, result.output
    assert "self-check" in result.output


def test_fit_with_holdout_on_tiny_dataset_warns_but_still_exits_zero(tmp_path):
    # Regression test: a real dataset too small to split into a holdout used to make the
    # self-check raise an unhandled ValueError, which crashed the whole `fit` command with a
    # traceback and a nonzero exit code, even though the profile had already been written
    # successfully by that point. The self-check failing should be a warning, not a crash.
    data_path = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [5.0]}).to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"

    result = runner.invoke(
        app, ["fit", str(data_path), "-o", str(profile_path), "--holdout", "0.2"]
    )

    assert result.exit_code == 0, result.output
    assert profile_path.exists()
    assert "self-check skipped" in result.output


def test_fit_constraints_flag_loads_and_enforces_a_yaml_file(tmp_path):
    # The README's own usage example shows --constraints as a first-class CLI flag, but nothing
    # exercised it through the actual command line: only parse_constraints() directly, and
    # Profile.fit(constraints=...) through the Python API. This wires it end-to-end.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "created_at": rng.integers(0, 1000, 500).astype(float),
            "updated_at": rng.integers(0, 1000, 500).astype(float),
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    constraints_path = tmp_path / "constraints.yaml"
    constraints_path.write_text(
        "constraints:\n"
        "  - type: inequality\n"
        "    left: created_at\n"
        '    op: "<="\n'
        "    right: updated_at\n"
    )

    profile_path = tmp_path / "profile.json"
    fit_result = runner.invoke(
        app,
        ["fit", str(data_path), "-o", str(profile_path), "--constraints", str(constraints_path)],
    )
    assert fit_result.exit_code == 0, fit_result.output

    output_path = tmp_path / "synthetic.csv"
    emit_result = runner.invoke(
        app, ["emit", str(profile_path), "-n", "300", "--seed", "0", "-o", str(output_path)]
    )
    assert emit_result.exit_code == 0, emit_result.output

    synthetic = pd.read_csv(output_path)
    assert (synthetic["created_at"] <= synthetic["updated_at"]).all()


def make_rating_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    # A small-integer column: type inference lands on categorical, which is often right for a
    # rating but wrong when the caller wants it modeled as a discrete numeric.
    return pd.DataFrame({"rating": rng.integers(1, 6, n), "amount": rng.normal(50, 10, n)})


def test_column_type_flag_overrides_inference(tmp_path):
    data_path = tmp_path / "data.csv"
    make_rating_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"

    result = runner.invoke(
        app,
        ["fit", str(data_path), "-o", str(profile_path), "--column-type", "rating=discrete"],
    )

    assert result.exit_code == 0, result.output
    assert Profile.load(profile_path).column_types["rating"] == "discrete"


def test_column_type_flag_is_repeatable(tmp_path):
    data_path = tmp_path / "data.csv"
    make_rating_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"

    result = runner.invoke(
        app,
        [
            "fit",
            str(data_path),
            "-o",
            str(profile_path),
            "--column-type",
            "rating=discrete",
            "--column-type",
            "amount=text",
        ],
    )

    assert result.exit_code == 0, result.output
    types = Profile.load(profile_path).column_types
    assert types["rating"] == "discrete"
    assert types["amount"] == "text"


def test_column_type_flag_rejects_an_unknown_type(tmp_path):
    data_path = tmp_path / "data.csv"
    make_rating_df().to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        [
            "fit",
            str(data_path),
            "-o",
            str(tmp_path / "profile.json"),
            "--column-type",
            "rating=nonsense",
        ],
    )

    assert result.exit_code != 0
    assert "unknown column type" in result.output


def test_column_type_flag_rejects_a_malformed_pair(tmp_path):
    data_path = tmp_path / "data.csv"
    make_rating_df().to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        ["fit", str(data_path), "-o", str(tmp_path / "profile.json"), "--column-type", "rating"],
    )

    assert result.exit_code != 0
    assert "NAME=TYPE" in result.output


def test_column_type_flag_rejects_a_column_not_in_the_dataset(tmp_path):
    data_path = tmp_path / "data.csv"
    make_rating_df().to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        [
            "fit",
            str(data_path),
            "-o",
            str(tmp_path / "profile.json"),
            "--column-type",
            "not_a_column=datetime",
        ],
    )

    assert result.exit_code != 0
    assert "not present in" in result.output


def test_emit_writes_synthetic_table(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    output_path = tmp_path / "synthetic.csv"
    result = runner.invoke(
        app, ["emit", str(profile_path), "-n", "50", "--seed", "1", "-o", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    synthetic = pd.read_csv(output_path)
    assert len(synthetic) == 50


def test_fit_and_emit_round_trip_through_parquet(tmp_path):
    # Every usage example in the README uses .parquet, not .csv, but no CLI test had actually
    # exercised that path end-to-end; the pyarrow-as-dev-only-dependency bug fixed earlier this
    # session would have made this exact fit -> emit sequence crash on a plain pip install.
    data_path = tmp_path / "data.parquet"
    make_df().to_parquet(data_path, index=False)
    profile_path = tmp_path / "profile.json"

    fit_result = runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])
    assert fit_result.exit_code == 0, fit_result.output

    output_path = tmp_path / "synthetic.parquet"
    emit_result = runner.invoke(
        app, ["emit", str(profile_path), "-n", "50", "--seed", "1", "-o", str(output_path)]
    )
    assert emit_result.exit_code == 0, emit_result.output

    synthetic = pd.read_parquet(output_path)
    assert len(synthetic) == 50


def test_check_passes_for_well_formed_fixtures(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df(n=2000).to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    fixtures_path = tmp_path / "fixtures.csv"
    runner.invoke(
        app, ["emit", str(profile_path), "-n", "2000", "--seed", "0", "-o", str(fixtures_path)]
    )

    result = runner.invoke(
        app,
        [
            "check",
            str(fixtures_path),
            "--profile",
            str(profile_path),
            "--real",
            str(data_path),
            "--min-dcr-ratio",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output


def test_check_reports_a_clean_error_instead_of_a_traceback_for_a_tiny_real_dataset(tmp_path):
    # Regression test: privacy.check() raises ValueError when the real dataset is too small to
    # split into a holdout and training set. fit --holdout already caught this as a warning;
    # the standalone check command let it propagate as a raw Python traceback instead.
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [5.0]}).to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    fixtures_path = tmp_path / "fixtures.csv"
    pd.DataFrame({"a": [1.0, 2.0]}).to_csv(fixtures_path, index=False)

    result = runner.invoke(
        app,
        ["check", str(fixtures_path), "--profile", str(profile_path), "--real", str(data_path)],
    )

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "check failed" in result.output
    assert "at least 2" in result.output


def test_check_fails_and_exits_nonzero_when_fixtures_are_real_data(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df(n=500).to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    result = runner.invoke(
        app,
        [
            "check",
            str(data_path),
            "--profile",
            str(profile_path),
            "--real",
            str(data_path),
            "--min-dcr-ratio",
            "1.0",
        ],
    )

    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_check_rare_combination_threshold_flag_changes_the_result(tmp_path):
    # Regression test: --rare-combination-threshold used to be parsed but never passed to
    # privacy.check(), so it had zero effect regardless of what the user set it to.
    rng = np.random.default_rng(0)
    n = 500
    region = np.array(rng.choice(["north", "south"], size=n, p=[0.5, 0.5]), dtype=object)
    plan_tier = np.array(rng.choice(["basic", "pro"], size=n, p=[0.5, 0.5]), dtype=object)
    region[:3] = "north"
    plan_tier[:3] = "enterprise-rare"  # a (north, enterprise-rare) combo occurring only 3 times

    df = pd.DataFrame({"region": region, "plan_tier": plan_tier})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    # Using the real data as its own "fixtures" guarantees the rare combo is reproduced, so
    # this isolates whether the threshold flag is honored rather than depending on sampling.
    base_args = [
        "check",
        str(data_path),
        "--profile",
        str(profile_path),
        "--real",
        str(data_path),
        "--min-dcr-ratio",
        "0",
    ]

    default_result = runner.invoke(app, base_args)
    lenient_result = runner.invoke(app, [*base_args, "--rare-combination-threshold", "1"])

    default_leaks = int(re.search(r"rare_combination_leaks: (\d+)", default_result.output).group(1))
    lenient_leaks = int(re.search(r"rare_combination_leaks: (\d+)", lenient_result.output).group(1))

    assert default_leaks > 0
    assert lenient_leaks == 0


def test_diff_passes_when_no_drift(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df(seed=0).to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    fresh_path = tmp_path / "fresh.csv"
    make_df(seed=1).to_csv(fresh_path, index=False)

    result = runner.invoke(app, ["diff", str(profile_path), str(fresh_path)])

    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output


def test_diff_fails_and_exits_nonzero_on_drift(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df(seed=0).to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    drifted = make_df(seed=1)
    drifted["amount"] = drifted["amount"] + 500
    drifted_path = tmp_path / "drifted.csv"
    drifted.to_csv(drifted_path, index=False)

    result = runner.invoke(app, ["diff", str(profile_path), str(drifted_path)])

    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_inspect_prints_a_summary_for_every_column(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    result = runner.invoke(app, ["inspect", str(profile_path)])

    assert result.exit_code == 0, result.output
    assert "amount" in result.output
    assert "plan_tier" in result.output
    assert "columns" in result.output


def test_inspect_reports_a_column_s_null_rate(tmp_path):
    df = make_df(n=500)
    df.loc[:99, "amount"] = None  # 20% null
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    result = runner.invoke(app, ["inspect", str(profile_path)])

    assert result.exit_code == 0, result.output
    assert "null_rate=0.200" in result.output


def test_compare_reports_added_removed_and_changed_columns(tmp_path):
    old_path = tmp_path / "old.csv"
    pd.DataFrame({"amount": make_df()["amount"], "gone": make_df()["amount"]}).to_csv(
        old_path, index=False
    )
    new_path = tmp_path / "new.csv"
    pd.DataFrame({"amount": make_df()["amount"] + 500, "fresh": make_df()["amount"]}).to_csv(
        new_path, index=False
    )

    old_profile = tmp_path / "old.json"
    new_profile = tmp_path / "new.json"
    runner.invoke(app, ["fit", str(old_path), "-o", str(old_profile)])
    runner.invoke(app, ["fit", str(new_path), "-o", str(new_profile)])

    result = runner.invoke(app, ["compare", str(old_profile), str(new_profile)])

    assert result.exit_code == 0, result.output
    assert "added:" in result.output
    assert "fresh" in result.output
    assert "removed:" in result.output
    assert "gone" in result.output
    assert "changed: amount" in result.output


def test_compare_says_so_when_nothing_changed(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    result = runner.invoke(app, ["compare", str(profile_path), str(profile_path)])

    assert result.exit_code == 0, result.output
    assert "no column-level changes" in result.output


def test_commands_reject_a_missing_input_file_without_a_traceback(tmp_path):
    # Regression test: a path typo is the most common CLI mistake there is, and every command
    # used to answer it with a full Python traceback from pandas or pathlib internals rather
    # than saying the file wasn't there.
    missing = tmp_path / "does_not_exist.json"
    missing_data = tmp_path / "does_not_exist.csv"
    out = tmp_path / "out.csv"

    invocations = [
        ["inspect", str(missing)],
        ["compare", str(missing), str(missing)],
        ["fit", str(missing_data), "-o", str(out)],
        ["emit", str(missing), "-n", "5", "--seed", "0", "-o", str(out)],
        ["diff", str(missing), str(missing_data)],
    ]

    for args in invocations:
        result = runner.invoke(app, args)
        assert result.exit_code != 0, args
        assert "does not exist" in result.output, args
        assert "Traceback" not in result.output, args


def test_check_rejects_missing_profile_and_real_files(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df(n=100).to_csv(data_path, index=False)
    missing = tmp_path / "gone.json"

    result = runner.invoke(
        app, ["check", str(data_path), "--profile", str(missing), "--real", str(data_path)]
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_fit_rejects_a_missing_constraints_file(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df(n=100).to_csv(data_path, index=False)

    result = runner.invoke(
        app,
        [
            "fit",
            str(data_path),
            "-o",
            str(tmp_path / "profile.json"),
            "--constraints",
            str(tmp_path / "missing.yaml"),
        ],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_inspect_json_flag_emits_parseable_json(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    result = runner.invoke(app, ["inspect", str(profile_path), "--json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["columns"] == 2
    assert parsed["rows_fit"] == 1000
    names = [c["name"] for c in parsed["column_summaries"]]
    assert names == ["amount", "plan_tier"]


def test_inspect_json_reports_null_rate_as_null_when_absent(tmp_path):
    data_path = tmp_path / "data.csv"
    make_df().to_csv(data_path, index=False)
    profile_path = tmp_path / "profile.json"
    runner.invoke(app, ["fit", str(data_path), "-o", str(profile_path)])

    parsed = json.loads(runner.invoke(app, ["inspect", str(profile_path), "--json"]).output)
    amount = next(c for c in parsed["column_summaries"] if c["name"] == "amount")
    assert amount["null_rate"] is None
    assert amount["in_copula"] is True


def _fit_two_profiles(tmp_path, shift=0.0):
    """Fit an 'old' and a 'new' profile; a nonzero shift makes them genuinely differ."""
    old_path, new_path = tmp_path / "old.csv", tmp_path / "new.csv"
    make_df().to_csv(old_path, index=False)
    shifted = make_df()
    shifted["amount"] = shifted["amount"] + shift
    shifted.to_csv(new_path, index=False)

    old_profile, new_profile = tmp_path / "old.json", tmp_path / "new.json"
    runner.invoke(app, ["fit", str(old_path), "-o", str(old_profile)])
    runner.invoke(app, ["fit", str(new_path), "-o", str(new_profile)])
    return old_profile, new_profile


def test_compare_json_flag_emits_parseable_json(tmp_path):
    old_profile, new_profile = _fit_two_profiles(tmp_path, shift=500.0)

    result = runner.invoke(app, ["compare", str(old_profile), str(new_profile), "--json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["rows_fit"] == [1000, 1000]
    assert [c["name"] for c in parsed["changed"]] == ["amount"]


def test_compare_fail_on_change_exits_nonzero_when_something_changed(tmp_path):
    old_profile, new_profile = _fit_two_profiles(tmp_path, shift=500.0)

    result = runner.invoke(app, ["compare", str(old_profile), str(new_profile), "--fail-on-change"])

    assert result.exit_code == 1


def test_compare_fail_on_change_exits_zero_when_nothing_changed(tmp_path):
    old_profile, _ = _fit_two_profiles(tmp_path)

    result = runner.invoke(app, ["compare", str(old_profile), str(old_profile), "--fail-on-change"])

    assert result.exit_code == 0
    assert "no column-level changes" in result.output


def test_compare_json_and_fail_on_change_combine(tmp_path):
    # --json returns early, so the exit code has to be applied on that path too.
    old_profile, new_profile = _fit_two_profiles(tmp_path, shift=500.0)

    result = runner.invoke(
        app, ["compare", str(old_profile), str(new_profile), "--json", "--fail-on-change"]
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["changed"]


def _fit_profile_with_foreign_key(tmp_path):
    data_path = tmp_path / "orders.csv"
    pd.DataFrame(
        {"customer_id": [7] * 200, "amount": np.random.default_rng(0).normal(50, 10, 200)}
    ).to_csv(data_path, index=False)

    constraints_path = tmp_path / "constraints.yaml"
    constraints_path.write_text(
        "constraints:\n"
        "  - type: foreign_key\n"
        "    column: customer_id\n"
        "    references: customers.id\n"
    )

    profile_path = tmp_path / "profile.json"
    runner.invoke(
        app,
        ["fit", str(data_path), "-o", str(profile_path), "--constraints", str(constraints_path)],
    )
    return profile_path


def test_emit_key_pool_draws_foreign_key_values_from_the_pool(tmp_path):
    # A foreign_key constraint was declarable in YAML but inert through the CLI: emit() takes
    # key_pools, the CLI never passed any, so the repair engine had nothing to sample and left
    # the column exactly as generated.
    profile_path = _fit_profile_with_foreign_key(tmp_path)
    customers_path = tmp_path / "customers.csv"
    pd.DataFrame({"id": [101, 102, 103], "name": list("abc")}).to_csv(customers_path, index=False)

    output_path = tmp_path / "out.csv"
    result = runner.invoke(
        app,
        [
            "emit",
            str(profile_path),
            "-n",
            "50",
            "--seed",
            "0",
            "-o",
            str(output_path),
            "--key-pool",
            f"customer_id={customers_path}:id",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = set(pd.read_csv(output_path)["customer_id"])
    assert emitted.issubset({101, 102, 103})
    assert emitted != {7}


def test_emit_warns_when_a_foreign_key_has_no_pool(tmp_path):
    profile_path = _fit_profile_with_foreign_key(tmp_path)
    result = runner.invoke(
        app,
        ["emit", str(profile_path), "-n", "5", "--seed", "0", "-o", str(tmp_path / "out.csv")],
    )

    assert result.exit_code == 0, result.output
    assert "no --key-pool given for foreign_key column(s)" in result.output


def test_emit_key_pool_rejects_a_malformed_spec(tmp_path):
    profile_path = _fit_profile_with_foreign_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "emit",
            str(profile_path),
            "-n",
            "5",
            "--seed",
            "0",
            "-o",
            str(tmp_path / "out.csv"),
            "--key-pool",
            "customer_id",
        ],
    )
    assert result.exit_code != 0
    assert "NAME=FILE:COLUMN" in result.output


def test_emit_key_pool_rejects_a_missing_file(tmp_path):
    profile_path = _fit_profile_with_foreign_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "emit",
            str(profile_path),
            "-n",
            "5",
            "--seed",
            "0",
            "-o",
            str(tmp_path / "out.csv"),
            "--key-pool",
            f"customer_id={tmp_path / 'nope.csv'}:id",
        ],
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_emit_key_pool_rejects_a_column_not_in_the_pool_file(tmp_path):
    profile_path = _fit_profile_with_foreign_key(tmp_path)
    customers_path = tmp_path / "customers.csv"
    pd.DataFrame({"id": [1, 2]}).to_csv(customers_path, index=False)

    result = runner.invoke(
        app,
        [
            "emit",
            str(profile_path),
            "-n",
            "5",
            "--seed",
            "0",
            "-o",
            str(tmp_path / "out.csv"),
            "--key-pool",
            f"customer_id={customers_path}:nope",
        ],
    )
    assert result.exit_code != 0
    assert "not in" in result.output
