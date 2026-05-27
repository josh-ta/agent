"""Tests for read-only Postgres query helpers."""

from __future__ import annotations

import pytest

from agent.memory.postgres_query import format_rows, validate_readonly_sql


def test_validate_readonly_sql_accepts_select_and_with() -> None:
    assert validate_readonly_sql("SELECT 1") == "SELECT 1"
    assert validate_readonly_sql("WITH x AS (SELECT 1) SELECT * FROM x;") == "WITH x AS (SELECT 1) SELECT * FROM x"


def test_validate_readonly_sql_rejects_mutating_and_multi_statement() -> None:
    with pytest.raises(ValueError, match="Only SELECT"):
        validate_readonly_sql("UPDATE events SET name='x'")
    with pytest.raises(ValueError, match="one SQL statement"):
        validate_readonly_sql("SELECT 1; SELECT 2")
    with pytest.raises(ValueError, match="Mutating"):
        validate_readonly_sql("SELECT * FROM events WHERE id IN (DELETE FROM events RETURNING id)")


def test_format_rows_table_and_csv() -> None:
    rows = [{"id": 1, "name": "alpha"}, {"id": 2, "name": None}]
    table = format_rows(rows, output_format="table")
    assert "id | name" in table
    assert "alpha" in table

    csv_out = format_rows(rows, output_format="csv")
    assert csv_out.splitlines()[0] == "id,name"
    assert "1,alpha" in csv_out
    assert "2," in csv_out


def test_format_rows_empty() -> None:
    assert format_rows([], output_format="table") == "(no rows)"
