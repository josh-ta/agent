# Communicate with Other Agents

Use this skill when you need to delegate a task to, or receive work from, another agent.

## Sending Progress Updates (for long tasks)

For any task expected to take more than 2 tool calls, send progress updates to the
user's channel so they know you're working. The loop automatically forwards `task_note()`
calls to Discord, so **the best approach is to call `task_note()` frequently** — you get
journal checkpointing AND Discord visibility in one call.

For step-by-step notes (auto-forwarded to Discord AND saved to journal):
```
task_note("Checked CI run 23065824301. Jobs: backend=failed. Root cause: PYTHONPATH missing. Next: fix ci-cd.yml.")
```

Use `task_note()` before the first tool call, after each major step, and again any time the task would otherwise go quiet for a while. Each note should cover:
- what you just did
- what you learned
- what you will do next

Do NOT call `send_discord` at the end to summarize — your final text response is the reply.

## Channel Layout

| Channel | Purpose |
|---|---|
| `#agent-<name>` | Each agent's private channel — streaming, reasoning, direct conversation |
| `#agent-comms` | Structured JSON task routing between agents (machine-readable) |
| `#agent-bus` | Brief status announcements to all agents |

**Only post to agent-comms and agent-bus.** Your thinking, tool calls, and progress
streaming automatically appear in your private channel — never in comms or bus.
`#agent-comms` is raw JSON only: no plain-text replies, no queue notices, and no receipt acks.

## Sending a Task to Another Agent

Post a JSON message to `#agent-comms` using the channel ID from your system prompt:
```
send_discord(AGENT_COMMS_CHANNEL_ID, '{"from": "YOUR_NAME", "to": "barbara", "task": "Fix the frontend ESLint errors in /workspace/TicketActionApp/frontend", "payload": ""}')
```

The channel IDs are listed in your system prompt under **Discord Channels** — use those exact numbers.

Then poll for their reply:
```
read_discord(DISCORD_COMMS_CHANNEL_ID, limit=10)
```
Look for a message with `"from": "agent-2"` and `"task": "result"`. The `"payload"` field
contains their answer. Do not expect a separate queued/received ack in `#agent-comms`, and do not send one back.

## Receiving an A2A Task (when another agent delegates to you)

When you receive a task prefixed `[A2A from X]`, another agent has delegated work to you.

1. Complete the task normally
2. When done, send your result back to comms so the requesting agent can read it:

```
send_discord(DISCORD_COMMS_CHANNEL_ID, '{"from": "YOUR_NAME", "to": "X", "task": "result", "payload": "your answer here"}')
```

Do NOT reply with plain text to agent-comms — the other agent reads structured JSON.
The only messages there should be task JSON and final result JSON. Never send ack, acknowledge, thank-you, or status-response messages there.

## Collaborating on a Shared Task

When a user asks two agents to collaborate:

**Agent 1 (coordinator):**
1. Send a JSON task to agent-2 via comms
2. Work on your portion
3. Poll comms every few tool calls: `read_discord(DISCORD_COMMS_CHANNEL_ID, limit=10)`
4. Incorporate agent-2's result into your final response

**Agent 2 (worker):**
1. Pick up the `[A2A from agent-1]` task automatically
2. Complete your portion
3. Send the result back to comms as JSON (see above)

## Broadcasting to All Agents

Use `"to": "*"` in the JSON payload:
```json
{"from": "agent-1", "to": "*", "task": "Update your GOALS.md with new objective: X"}
```

## Monitoring the Bus

Read recent messages from `#agent-bus`:
```
read_discord(DISCORD_BUS_CHANNEL_ID, limit=20)
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
  "task": "what to do  (use 'result' when replying with output)",
  "payload": "task details or result content (optional)"
}
```

## Tips

- Keep messages concise — Discord has a 2,000 char limit.
- For large payloads, write to a shared file in `/workspace` and reference the path.
- Always check `task_resume()` at startup to process any pending delegated tasks.
- Be explicit in task descriptions: who should do what by when.
