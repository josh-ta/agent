# Communicate with Other Agents

Use this skill when you need to delegate a task to, or receive work from, another agent.

## Channel Layout

| Channel | Purpose |
|---|---|
| `#agent-bus` | Announcements and informal broadcasts to all agents |
| `#agent-comms` | Structured JSON task routing (machine-readable) |
| `#agent-<name>` | Each agent's private channel for direct tasks and logs |

## Sending a Task to Another Agent

Post a JSON message to `#agent-comms`:
```
send_discord(DISCORD_COMMS_CHANNEL_ID, '{"from": "agent-1", "to": "agent-2", "task": "Research X and return a summary", "payload": ""}')
```

The receiving agent will pick this up automatically and reply in its own channel or back to comms.

## Monitoring the Bus

Read recent messages from `#agent-bus`:
```
read_discord(DISCORD_BUS_CHANNEL_ID, limit=20)
```

## Broadcasting to All Agents

Use `"to": "*"` in the JSON payload:
```json
{"from": "agent-1", "to": "*", "task": "Update your GOALS.md with new objective: X"}
```

## Checking Agent Status

If Postgres is enabled:
```
list_agents()
```

## A2A Message Format

```json
{
  "from": "agent-name",
  "to": "target-agent-name or *",
  "task": "Human-readable task description",
  "payload": "Additional context or data (optional)"
}
```

## Tips

- Keep messages concise — Discord has a 2,000 char limit.
- For large payloads, write to a shared file in `/workspace` and reference the path.
- Always check `my_tasks()` at startup to process any pending delegated tasks.
- Be explicit in task descriptions: who should do what by when.
