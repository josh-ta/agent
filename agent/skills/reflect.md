# Reflect and Learn

Use this skill after completing or failing a task to extract lessons, fix mistakes, and improve yourself.

## When to Reflect

- **After any failure**: Always reflect. Understand why it failed and record what to do differently.
- **After a novel task**: If you learned something reusable, save it.
- **After repeated tool attempts**: If you tried something multiple times, save the pattern that worked.
- **During heartbeat** (every ~60s idle): Review recent mistakes and update MEMORY.md if needed.

## Procedure for Failures

1. **Diagnose**: What specifically went wrong? (wrong flag, missing dependency, bad assumption?)
2. **Record the mistake**:
   ```
   lesson_save("MISTAKE: <what failed> — instead do <correct approach>", kind="mistake")
   ```
3. **Update the relevant skill** if the mistake was procedural (e.g. wrong command syntax):
   ```
   skill_read("skill-name")   # read current content first
   skill_edit("skill-name", "<corrected content>")
   ```
4. **Update MEMORY.md** if the lesson is important long-term:
   ```
   identity_read("MEMORY.md")
   identity_edit("MEMORY.md", "<existing content>\n\n### Mistake — YYYY-MM-DD\n<lesson>")
   ```

## Procedure for Insights

When you discover something useful (a better approach, an API quirk, a pattern):
```
lesson_save("INSIGHT: <what you learned>", kind="insight")
memory_save("<fact to remember>")
```

## Reviewing Past Lessons Before a Task

Before starting a complex task, check for relevant past lessons:
```
lesson_search("<keywords from the task>")
```

If lessons are found, read them and adjust your approach accordingly.

## Periodic Self-Review

Run this every 10 tasks or when you have idle time:
```
lessons_recent(limit=20)
```
Identify any patterns in your mistakes and update the relevant skills to prevent recurrence.

## Rules

- Never record a lesson longer than 2 sentences — keep them actionable.
- A lesson must say what to do differently, not just what went wrong.
- Don't duplicate lessons — search first before saving.
