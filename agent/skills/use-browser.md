# Use the Browser

Use this skill when you need to navigate the web, fill forms, click buttons, or take screenshots.

## Tools Available

The browser is controlled via the Playwright MCP server running as a sidecar container.
You can watch the browser live at `http://<server>:6080` (noVNC).

Playwright MCP exposes tools like:
- `browser_navigate` — go to a URL
- `browser_click` — click an element
- `browser_type` / `browser_fill` — type text
- `browser_snapshot` — get page structure (accessibility tree)
- `browser_screenshot` — take a screenshot
- `browser_scroll` — scroll the page

## Procedure

1. **Navigate** to the target URL.
2. **Snapshot** the page to understand its structure.
3. **Interact**: click, fill, scroll as needed.
4. **Extract**: read text from the snapshot or screenshot.
5. **Close/navigate away** when done.

## Example: Fill a Search Form

```
browser_navigate(url="https://example.com")
browser_snapshot()    # understand the page structure
browser_fill(selector="input[name='q']", value="my search query")
browser_click(selector="button[type='submit']")
browser_snapshot()    # read the results
```

## Tips

- Always snapshot before clicking — element refs change after navigation.
- For login pages, credentials may be in `.env` or ask the user via Discord.
- If you encounter a CAPTCHA, alert the user in Discord and pause.
- Prefer the accessibility tree (snapshot) over screenshots for text extraction — it's faster and cheaper.
- For downloads, check the `/workspace/downloads` directory after triggering a download.
