# Query Postgres

Use when the user asks for data, reports, CSV exports, or SQL against the connected database (`POSTGRES_URL`).

## Workflow

1. **Discover schema** (if table/column names are unknown):
   ```
   list_postgres_tables()
   query_postgres("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' AND table_name='events' ORDER BY ordinal_position")
   ```

2. **Run the query** with `query_postgres()`. Only `SELECT`, `WITH`, `EXPLAIN`, or `TABLE` statements are allowed.

3. **CSV export** (write directly to disk — never paste CSV into `write_file`):
   ```
   query_postgres(
     "SELECT ...",
     output_format='csv',
     output_path='/workspace/upcoming_arena_events_ticket_limit_4.csv',
     limit=5000,
   )
   ```
   The file is attached to the Discord reply automatically when export succeeds.

## Tips

- Filter upcoming rows with `WHERE event_date >= CURRENT_DATE` (adjust column names after schema discovery).
- Start with `LIMIT 10` while iterating on the query shape.
- If results hit the row limit, tighten filters instead of raising limit blindly.
- Do not use `run_shell` + `psql` when `query_postgres` is available.

## Common mistakes

- Answering from memory instead of querying — always call tools first.
- Guessing table names — use `list_postgres_tables()` or `information_schema`.
- Piping large CSV results through `write_file` — use `output_path` on `query_postgres` instead.
