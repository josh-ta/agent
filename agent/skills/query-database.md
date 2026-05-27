# Query Postgres

Use when the user asks for data, reports, rankings, CSV exports, or SQL against the connected database (`POSTGRES_URL`).

## Analytics (default — answer in Discord)

For questions like "top 5 events", "sales starting today", "which should I buy":

1. **Schema** (only if you don't already know columns from this session):
   ```
   list_postgres_tables()
   query_postgres("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' AND table_name='events' ORDER BY ordinal_position LIMIT 50")
   ```

2. **Query** with `query_postgres()` — use `LIMIT` (10–50 for exploration, 5 for "top N" answers):
   ```
   query_postgres("SELECT name, event_date, sales_start, chartmetric_popularity_score, ... FROM events WHERE sales_start::date = CURRENT_DATE ORDER BY chartmetric_popularity_score DESC NULLS LAST LIMIT 5")
   ```

3. **Answer in prose** — summarize the rows for the user. Do not export CSV unless they ask for a file.

**Efficiency:** Aim for 1 schema check + 1–2 data queries. Do not loop more than 5 `query_postgres` calls.

## CSV export (only when user asks for a file)

```
query_postgres(
  "SELECT ...",
  output_format='csv',
  output_path='/workspace/export.csv',
  limit=5000,
)
```

The file is **uploaded to Discord automatically**. Do not tell the user to open `/workspace`.

## Tips

- Filter upcoming rows with `WHERE event_date >= CURRENT_DATE`.
- For "sale starting today": filter on `sales_start` (or `discovery_sale_start_at` / `pipeline_sale_start_at` after schema check).
- Rank by chartmetric scores, popularity, or qualification score when picking "top" events.
- Do not use `run_shell` + `psql` when `query_postgres` is available.

## Common mistakes

- Answering from memory instead of querying — always call tools first.
- Claiming you lack database access — you have `list_postgres_tables` and `query_postgres`.
- Exporting CSV when the user only asked a question.
- Running 10+ exploratory queries — tighten SQL instead.
