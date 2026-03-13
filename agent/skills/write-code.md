# Write Code

Use this skill when asked to write, debug, or refactor code in any language.

## Procedure

1. **Understand requirements**: Ask clarifying questions if the spec is ambiguous.
2. **Plan**: Think through the approach before writing. Consider edge cases.
3. **Write to workspace**: Create the file in `/workspace` using `write_file`.
   - Always include a header comment explaining purpose and usage.
   - Keep functions small and focused.
4. **Validate**: Run the code with `run_shell` to check for syntax errors and basic functionality.
   ```
   run_shell("python /workspace/myscript.py")
   ```
5. **Iterate**: Fix errors revealed by running the code. Re-run until clean.
6. **Test**: Write at least one smoke test; run it.
7. **Document**: Update or create a README in the same directory if non-trivial.

## Language-specific Tips

### Python
- Use f-strings, type hints, and dataclasses.
- Prefer `pathlib.Path` over `os.path`.
- Format with `ruff format` if available: `run_shell("ruff format /workspace/file.py")`

### Shell Scripts
- Start with `#!/usr/bin/env bash` and `set -euo pipefail`.
- Quote all variable expansions: `"$var"` not `$var`.

### JavaScript / TypeScript
- Use `async/await` over callbacks.
- Prefer `const` over `let`; avoid `var`.

## Error Handling
- If the code fails, read the full error message carefully.
- Search memory for similar past errors before escalating.
- Never guess at a fix without understanding the error.
