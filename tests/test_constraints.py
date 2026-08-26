import pytest

from synthkit.constraints import (
    ConditionalNull,
    Derived,
    ForeignKey,
    Inequality,
    Unique,
    constraints_to_dicts,
    parse_constraints,
)


def test_parse_inequality_from_dict_list():
    spec = [{"type": "inequality", "left": "created_at", "op": "<=", "right": "updated_at"}]
    constraints = parse_constraints(spec)
    assert constraints == [Inequality("created_at", "<=", "updated_at")]


def test_parse_all_constraint_types():
    spec = [
        {"type": "inequality", "left": "a", "op": "<=", "right": "b"},
        {"type": "derived", "column": "total", "expr": "subtotal + tax"},
        {
            "type": "conditional_null",
            "column": "cancelled_at",
            "null_when": "status != 'cancelled'",
        },
        {"type": "unique", "columns": ["customer_id"]},
        {"type": "foreign_key", "column": "customer_id", "references": "customers.id"},
    ]
    constraints = parse_constraints(spec)
    assert isinstance(constraints[0], Inequality)
    assert isinstance(constraints[1], Derived)
    assert isinstance(constraints[2], ConditionalNull)
    assert isinstance(constraints[3], Unique)
    assert isinstance(constraints[4], ForeignKey)


def test_parse_wraps_top_level_constraints_key():
    spec = {"constraints": [{"type": "unique", "columns": ["id"]}]}
    constraints = parse_constraints(spec)
    assert constraints == [Unique(["id"])]


def test_invalid_inequality_operator_rejected():
    with pytest.raises(ValueError):
        Inequality("a", "==", "b")


def test_unknown_constraint_type_rejected():
    with pytest.raises(ValueError):
        parse_constraints([{"type": "nonsense"}])


def test_parse_from_yaml_file(tmp_path):
    yaml_text = """
constraints:
  - type: inequality
    left: created_at
    op: "<="
    right: updated_at
  - type: unique
    columns: [customer_id]
"""
    path = tmp_path / "constraints.yaml"
    path.write_text(yaml_text)
    constraints = parse_constraints(path)
    assert constraints == [
        Inequality("created_at", "<=", "updated_at"),
        Unique(["customer_id"]),
    ]


def test_round_trip_through_dicts():
    original = [Inequality("a", "<=", "b"), Derived("total", "subtotal + tax")]
    dicts = constraints_to_dicts(original)
    restored = parse_constraints(dicts)
    assert restored == original
