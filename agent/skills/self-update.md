# Self-Update

Use this skill when you want to improve your own capabilities, fix a bug in your code,
add a new skill, or update your identity/goals.

## Levels of Self-Modification

### Level 1: Update Skills (fastest, no restart needed)
Edit or create `.md` files in `agent/skills/`. Changes are loaded on next task.
```
skill_edit("new-skill-name", "# New Skill\n\n## Procedure\n...")
```

### Level 2: Update Identity (no restart needed)
Edit `IDENTITY.md`, `GOALS.md`, or `MEMORY.md`.
```
identity_edit("GOALS.md", "# Goals\n\n1. ...\n")
```

### Level 3: Update Python Code (requires container restart)
1. Read the file you want to change:
   ```
   read_file("/app/agent/tools/shell.py")
   ```
2. Write the updated version:
   ```
   write_file("/app/agent/tools/shell.py", "<new content>")
   ```
3. Commit to git:
   ```
   run_shell("cd /app && git add -A && git commit -m 'agent: <description of change>'")
   ```
4. Restart the container:
   ```
   agent_restart("code update: <description>")
   ```
   ⚠️ You will be offline for ~15–30 seconds. Warn users in Discord first.

## Safety Rules

- **Never** modify files in `agent/core.py` in ways that remove safety constraints.
- **Always** read a file before writing it (to avoid overwriting with incomplete content).
- **Always** test Python changes with `run_shell("python -m py_compile <file>")` before committing.
- If a skill is no longer useful, delete it with `run_shell("rm /app/agent/skills/<name>.md")`.
- Document the reason for changes in the git commit message.

## Adding a New Skill

1. Identify the procedure to document.
2. Write the skill in Markdown with: title (h1), use-case, step-by-step procedure, tips.
3. Save it: `skill_edit("skill-name", content)`.
4. Confirm it appears: `skill_list()`.
