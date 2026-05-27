# Events table — schema reference

Canonical reference for the **`events`** table in Postgres. Use this before exploratory schema queries when column names are already listed here.

## Core identifiers & dates

| Column | Meaning |
|--------|---------|
| `name` | Event title |
| `event_date`, `event_time` | Show date/time |
| `sales_start`, `sales_end` | Public onsale window (primary filter for "sale starting today") |
| `discovery_sale_start_at`, `discovery_sale_type` | Early discovery / presale signals |
| `pipeline_sale_start_at`, `pipeline_sale_type` | Pipeline presale window |
| `status_code` | Sale status (e.g. onsale, rescheduled) |

## Venue & market

| Column | Meaning |
|--------|---------|
| `venue_name`, `city`, `state_code`, `country_code` | Location |
| `venue_kind` | e.g. `arena`, `stadium`, `theatre` — filter with `IN ('arena','stadium')` for large-cap |

## Tickets & pricing (snapshot)

| Column | Meaning |
|--------|---------|
| `ticket_limit_max` | Primary purchase limit (often 4, 6, 8) |
| `price_min`, `price_max`, `price_currency` | Primary face range when on sale — **not** a full price history |
| `classification_genre_name`, `classification_segment_name` | Genre/segment for hold vs spec judgment |

## Chartmetric (artist demand signals)

| Column | Meaning |
|--------|---------|
| `chartmetric_popularity_score` | Overall popularity — primary rank key |
| `chartmetric_momentum_score` | Recent momentum / hype |
| `chartmetric_listener_count` | Audience size proxy |
| `chartmetric_artist_rank` | Rank (lower = bigger) |
| `chartmetric_career_stage`, `chartmetric_career_trend` | Career phase / direction |

## Common filters

```sql
-- Upcoming shows
WHERE event_date >= CURRENT_DATE

-- Sale starting today
WHERE sales_start::date = CURRENT_DATE

-- Arena/stadium + ticket limit 4
WHERE venue_kind IN ('arena', 'stadium') AND ticket_limit_max = 4

-- Upcoming public sales
WHERE sales_start >= CURRENT_DATE OR sales_start IS NULL
```

## Price history note

The `events` row is a **snapshot**. Time-series secondary pricing may live in other tables (e.g. `price_snapshots`, `listing_history`) — discover with `list_postgres_tables()` when Josh asks about **price drops over time**. If no history table exists or event isn’t on sale yet, use proxy signals from `event-spec-analysis.md`.

## Query discipline

- Skip `information_schema` if this skill is already in context.
- Prefer 1–2 focused SELECTs with explicit columns over `SELECT *`.
- Always use `LIMIT` on exploration queries.
