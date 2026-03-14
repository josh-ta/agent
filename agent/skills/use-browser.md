# Use the Browser

Use this skill when you need to navigate the web, fill forms, click buttons, or take screenshots.

## Tools Available

The browser is controlled via the Playwright MCP server running as a sidecar container.
A human can watch the browser live at `http://<server>:6080` (noVNC viewer — read-only for humans).

| Tool | Purpose |
|---|---|
| `browser_navigate(url)` | Go to a URL |
| `browser_snapshot()` | Get the accessibility tree (structure + selectors) — preferred over screenshots for understanding layout |
| `browser_screenshot()` | Take a PNG screenshot — use when visual layout matters |
| `browser_content()` | Get page text as clean Markdown — best for reading articles, tables, listings |
| `browser_click(selector)` | Click an element by CSS selector |
| `browser_fill(selector, value)` | Clear a field and type new text |
| `browser_type(selector, text)` | Append text to a field (without clearing) |
| `browser_scroll(direction, amount)` | Scroll page (`direction`: down/up/left/right, `amount`: pixels) |
| `browser_scroll(selector=...)` | Scroll a specific element into view |
| `browser_evaluate(script)` | Execute JavaScript and return the result |

## Procedure

1. **Navigate** to the target URL with `browser_navigate(url="https://...")`.
2. **Snapshot** the page with `browser_snapshot()` to understand structure and find selectors.
3. **Interact**: `browser_fill`, `browser_click`, `browser_scroll` as needed.
4. **Extract**: use `browser_content()` for text, `browser_snapshot()` for structure, `browser_screenshot()` only when visual confirmation is needed.
5. **Navigate away** or close when done.

## Example: Fill a Search Form

```
browser_navigate(url="https://example.com")
browser_snapshot()                                    # understand page structure
browser_fill(selector="input[name='q']", value="my search query")
browser_click(selector="button[type='submit']")
browser_snapshot()                                    # read the results
browser_content()                                     # extract text
```

## Example: Scroll and Read a Long Page

```
browser_navigate(url="https://example.com/long-article")
browser_content()                                     # get above-the-fold text
browser_scroll(direction="down", amount=1000)
browser_content()                                     # get rest of page
```

## Tips

- Always snapshot before clicking — element refs change after navigation.
- Prefer `browser_snapshot()` + `browser_content()` over `browser_screenshot()` for text — faster and cheaper.
- For login pages, credentials may be in `.env` or ask the user via `ask_user_question`.
- If you encounter a CAPTCHA, alert the user in Discord and pause.
- For downloads, check `/workspace/downloads` after triggering a download.
- The browser persists between tool calls in the same session — no need to re-navigate unless the page changes.
