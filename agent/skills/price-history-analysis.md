# Price history analysis

Use when Josh asks about **price drops over time**, **historical pricing**, **how much an event dropped since onsale**, or when **`price_min` / `price_max` snapshots aren’t enough** and time-series data may exist.

## When to use which approach

| Situation | Approach |
|-----------|----------|
| Event **not on sale yet** | No row-level price history — use **`event-spec-analysis`** proxy signals; do not refuse |
| Event on sale, only snapshot columns on `events` | Compare `price_min`/`price_max` to face value; note snapshot limitation |
| **History tables exist** (discover via `list_postgres_tables()`) | Query time series; compute drop %, days-to-low, trend |

## Discover history tables

```
list_postgres_tables()
```

Look for names like: `price_snapshots`, `listing_prices`, `event_prices`, `secondary_prices`, `inventory_snapshots`. Then inspect columns:

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = '<table>'
ORDER BY ordinal_position
```

Typical join key: `event_id` or match on `name` + `event_date` (prefer event_id).

## Analysis patterns

**Drop since onsale** (adapt table/column names):

```sql
SELECT e.name, e.event_date, e.sales_start,
       MIN(p.price) AS low_price,
       MAX(p.price) AS high_price,
       MAX(p.price) - MIN(p.price) AS range_drop,
       COUNT(*) AS snapshot_count
FROM events e
JOIN price_snapshots p ON p.event_id = e.id
WHERE e.event_date >= CURRENT_DATE
GROUP BY e.id, e.name, e.event_date, e.sales_start
HAVING COUNT(*) >= 2
ORDER BY range_drop DESC NULLS LAST
LIMIT 20
```

**Percent drop from peak** (when columns support it):

```sql
-- (peak - current) / peak — adjust column names to match schema
```

## If history is sparse or empty

1. Say how many snapshots you found (0 is valid).
2. **Pivot to proxy analysis** using chartmetric, venue, runway, limits — same as spec skill.
3. Deliver ranked recommendations anyway — never stop at "no historical data."

## Answer format

- **Data coverage**: N events with history, date range of snapshots
- **Top movers**: events with largest drop (or hold) with numbers
- **Recommendation**: tie back to Josh’s ask (spec vs hold vs buy today)
- If falling back to proxies: **"Method: limited history — ranking uses …"**

## Avoid

- "I cannot fulfill without historical pricing" as a final answer
- Running 10+ schema probes — pick the most likely table and validate once
