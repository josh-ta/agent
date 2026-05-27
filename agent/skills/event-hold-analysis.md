# Event hold analysis — sticky demand / won’t drop much

Use when Josh asks which events to **hold** (expect prices to **stay firm** or **not drop much** on secondary), which inventory is **safe to keep**, or which picks are **bad specs** because demand is sticky.

## Mindset

- Opposite bias from `event-spec-analysis`: rank events likely to **hold floor** or **appreciate**.
- Query Postgres, then **rank with reasoning** — do not ask for criteria.
- Missing price history is OK — use proxy signals below.

## Signals — rank HIGHER for hold (lower drop risk)

| Signal | Why |
|--------|-----|
| **Superstar / low chartmetric_artist_rank** (top tier) | Fan intensity; secondary holds |
| **Short runway** (event within 30–45 days of sale) | Less time for inventory to build |
| **Strong career_trend** + high popularity | Growing, not fading hype |
| **Legacy / catalog acts**, strong genre loyalty | Sticky demand (many country, classic rock) |
| **Small venue / theatre** vs oversized arena | Supply/demand tighter |
| **Low ticket_limit_max** on hyped act | Scarcity supports floor |
| **Single major event in market that week** | No competing demand split |

## Signals — rank LOWER for hold (likely to drop — better spec)

| Signal | Why |
|--------|-----|
| Long sale→event runway (90+ days) | Time for secondary to soften |
| Mid-tier popularity in huge arena | Oversupply risk |
| Fading momentum, trendy genre | Hype decay |
| Many competing events same city/week | See `market-competition` patterns in spec skill |

## Workflow

1. Query candidates (upcoming or on-sale events Josh scoped):
   ```sql
   SELECT name, event_date, sales_start, venue_name, city, venue_kind,
          ticket_limit_max, price_min, price_max,
          chartmetric_popularity_score, chartmetric_momentum_score,
          chartmetric_artist_rank, chartmetric_career_stage, chartmetric_career_trend,
          classification_genre_name
   FROM events
   WHERE event_date >= CURRENT_DATE
   ORDER BY chartmetric_popularity_score DESC NULLS LAST
   LIMIT 80
   ```

2. Rank top N **hold candidates** (or “avoid spec” list if asked).

## Answer format

Numbered list:

1. **Event** — venue, dates  
   **Hold case**: 2 bullets  
   **Spec risk**: one line if relevant  

End with **Method** — one sentence on signals used.

## Avoid

- Confusing hold with spec (hold = won’t drop much; spec = expect drop)
- Refusing due to missing historical prices
