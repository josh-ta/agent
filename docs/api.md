# API Reference

The control plane is a minimal FastAPI surface for health checks, task submission, task lookup, and server-sent task events.

## Generated docs

- Swagger UI: `/docs`
- OpenAPI schema: `/openapi.json`

When running locally with the default settings, those are:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

## Endpoints

### `GET /healthz`

Liveness probe.

Response:

```json
{"status":"ok"}
```

### `GET /readyz`

Readiness probe for the control-plane runtime. Returns `503` with a typed error response if the runtime has not finished starting or has failed.

### `POST /tasks`

Submit a task for the agent to process.

Request:

```json
{
  "content": "Summarize the repository",
  "author": "api-client",
  "metadata": {
    "request_id": "demo-123"
  }
}
```

Response:

```json
{
  "id": "9b3a8d1d-2a9d-4a95-9148-66ea5d3a7f1c",
  "status": "queued"
}
```

### `GET /tasks/{id}`

Look up the persisted state of a task in SQLite. Task states are `queued`, `running`, `succeeded`, and `failed`.

### `GET /events`

Server-sent events stream for task activity.

- Pass `task_id=<id>` to watch one task.
- Omit `task_id` to watch all task-tagged events.

Example:

```bash
curl -N "http://127.0.0.1:8000/events?task_id=<task-id>"
```

## Error model

Errors share one typed envelope:

```json
{
  "error_code": "task_not_found",
  "message": "Task 'missing' was not found.",
  "details": null
}
```

Stable `error_code` values:

- `invalid_request`
- `task_not_found`
- `service_unavailable`
- `internal_error`
