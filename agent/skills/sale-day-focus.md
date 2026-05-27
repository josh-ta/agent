# Sale-day focus — what to buy today

Use when Josh asks which events to **focus on buying** with a **sale starting today** (or onsale today / going on sale now). This is **execution priority**, not long-term spec.

## Mindset

- **Query first**, then deliver a ranked buy list (usually top 5–10). Do not ask for criteria.
- Today’s onsale = **time-sensitive** — prioritize chartmetric demand, venue size, limits, and sale window clarity.
- You are picking **where to spend attention right now**, not predicting months-ahead price drops (see `event-spec-analysis` for that).

## Workflow

1. **Pull today’s sales** (adjust date column after schema check if needed):
   ```sql
   SELECT name, event_date, event_time, sales_start, sales_end,
          venue_name, city, state_code, venue_kind,
          ticket_limit_max, price_min, price_max, status_code,
          chartmetric_popularity_score, chartmetric_momentum_score,
          chartmetric_artist_rank, chartmetric_career_stage,
          classification_genre_name, classification_segment_name
   FROM events
   WHERE sales_start::date = CURRENT_DATE
     AND event_date >= CURRENT_DATE
   ORDER BY chartmetric_popularity_score DESC NULLS LAST,
            chartmetric_momentum_score DESC NULLS LAST
   LIMIT 50
   ```

2. **Rank for buy focus** — weight heavily:
   - High **chartmetric_popularity_score** + **momentum** (demand likely strong at onsale)
   - **Arena/stadium** + major market (volume opportunity)
   - Clear **ticket_limit_max** (often 4 or 6 — note it)
   - **Genre/segment** with strong secondary liquidity (pop, country, legacy rock — use judgment)
   - Avoid spreading thin: prefer 5 strong picks over 15 weak ones

3. **Optional**: same-city events this week (competition dilutes focus):
   ```sql
   SELECT city, COUNT(*) AS cnt
   FROM events
   WHERE event_date >= CURRENT_DATE
     AND event_date < CURRENT_DATE + INTERVAL '7 days'
   GROUP BY 1 HAVING COUNT(*) >= 3
   ORDER BY cnt DESC LIMIT 15
   ```

## Answer format

Numbered list — for each event:

1. **Event name** — venue, city | sale starts [time if known] | event date  
   **Why buy focus**: 2 bullets from query data  
   **Pass if**: one line (e.g. low popularity, club show, unclear sale time)

Close with: **"Today’s pick order"** — one sentence on how you ranked.

## Avoid

- Asking Josh to define "focus on buying"
- Claiming no database access
- Spec/price-drop framing unless he asked for that
