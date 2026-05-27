"""Read-only SQL helpers for the configured Postgres connection."""

from __future__ import annotations

import csv
import io
import re
from typing import Any, Literal

_MUTATING = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|"
    r"merge|call|execute|vacuum|reindex|cluster|comment|refresh|security|"
    r"lock|notify|listen|set\s+role|reset\s+role"
    r")\b",
    re.IGNORECASE,
)


def validate_readonly_sql(sql: str) -> str:
    """Return normalized SQL if it is a single read-only statement."""
    text = sql.strip()
    if not text:
        raise ValueError("SQL query is empty")

    # Strip trailing semicolon only; reject embedded statement separators.
    if ";" in text.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed per call")

    normalized = text.rstrip(";").strip()
    head = normalized.lstrip("(").lstrip()
    if not head.upper().startswith(("SELECT", "WITH", "EXPLAIN", "TABLE")):
        raise ValueError("Only SELECT/WITH/EXPLAIN/TABLE read queries are allowed")

    if _MUTATING.search(normalized):
        raise ValueError("Mutating SQL keywords are not allowed")

    return normalized


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)


def format_rows(rows: list[dict[str, Any]], *, output_format: Literal["table", "csv"] = "table") -> str:
    if not rows:
        return "(no rows)"

    columns = list(rows[0].keys())
    if output_format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(col)) for col in columns])
        return buf.getvalue().rstrip()

    header = " | ".join(columns)
    sep = "-+-".join("-" * max(3, len(col)) for col in columns)
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(_cell(row.get(col)) for col in columns))
    return "\n".join(lines)
