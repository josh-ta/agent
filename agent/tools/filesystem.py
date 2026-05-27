"""
Filesystem tools: read, write, list, delete files in or alongside the workspace.

Relative paths resolve under WORKSPACE_PATH. When FILESYSTEM_STRICT_WORKSPACE=true,
all paths must stay inside the workspace (CC-style). When false, paths outside the
workspace are allowed but logged so operational use of e.g. /tmp remains visible.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import structlog

from agent.config import settings

log = structlog.get_logger()

MAX_READ_BYTES = 32 * 1024  # 32 KB for whole-file reads
MAX_LINE_WINDOW = 2000  # max lines returned per read_file line-range call


def _workspace_root() -> Path:
    return settings.workspace_path.resolve()


def _strict_workspace_violation(resolved: Path) -> str | None:
    """If strict mode is on and path is outside workspace, return error message."""
    if not settings.filesystem_strict_workspace:
        return None
    root = _workspace_root()
    try:
        resolved.relative_to(root)
    except ValueError:
        return (
            f"[ERROR: path outside workspace (FILESYSTEM_STRICT_WORKSPACE=true): {resolved}. "
            f"Workspace root: {root}]"
        )
    return None


def _safe_path(path: str) -> Path:
    """
    Resolve path under workspace when relative; absolute paths are resolved as-is.
    Logs (or rejects in strict mode) when the result lies outside the workspace.
    """
    workspace = _workspace_root()
    raw = Path(path)
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (workspace / path).resolve()

    err = _strict_workspace_violation(resolved)
    if err:
        raise PermissionError(err)

    if not resolved.is_relative_to(workspace):
        log.warning("path_outside_workspace", path=path, resolved=str(resolved))

    return resolved


def read_file(
    path: str,
    encoding: str = "utf-8",
    *,
    start_line: int = 1,
    end_line: int | None = None,
    max_lines: int = MAX_LINE_WINDOW,
) -> str:
    """
    Read a file and return its content.

    Whole-file mode (default): start_line==1 and end_line is None — reads the full
    file, capped by MAX_READ_BYTES (binary mode unchanged).

    Line-window mode: set end_line and/or start_line>1 — streams only that range,
    capped at max_lines lines, then applies MAX_READ_BYTES on the slice. Line numbers
    in the output are 1-based and prefixed for navigation.

    Args:
        path: File path (relative to workspace or absolute).
        encoding: Text encoding. Use 'binary' to get hex dump of binary files.
        start_line: First line to include (1-based). Ignored for encoding=='binary'.
        end_line: Last line to include (1-based), inclusive; None means end of file
            in line mode, or full file when start_line==1 (legacy whole-file read).
        max_lines: Safety cap on how many lines to return in line-window mode.

    Returns:
        File content as string, or error message.
    """
    try:
        fp = _safe_path(path)
        if not fp.exists():
            return f"[ERROR: file not found: {path}]"
        if not fp.is_file():
            return f"[ERROR: not a file: {path}]"

        size = fp.stat().st_size
        if encoding == "binary":
            data = fp.read_bytes()
            return data[:MAX_READ_BYTES].hex()

        # Legacy: full file with byte cap
        if start_line == 1 and end_line is None:
            content = fp.read_text(encoding=encoding, errors="replace")
            if len(content) > MAX_READ_BYTES:
                content = content[:MAX_READ_BYTES] + (
                    f"\n... [truncated, total {size} bytes — use start_line/end_line "
                    "to read a window]"
                )
            return content

        sl = max(1, int(start_line))
        el = int(end_line) if end_line is not None else None
        cap = max(1, min(int(max_lines), MAX_LINE_WINDOW))

        lines_out: list[str] = []
        current = 0
        with fp.open(encoding=encoding, errors="replace") as handle:
            for line in handle:
                current += 1
                if current < sl:
                    continue
                if el is not None and current > el:
                    break
                lines_out.append(f"{current:6d}|{line}")
                if len(lines_out) >= cap:
                    lines_out.append(
                        f"... [max_lines={cap} reached; narrow range or raise max_lines]"
                    )
                    break

        if el is not None and current < el and len(lines_out) < cap:
            pass  # EOF before end_line — fine

        if not lines_out and sl > 1:
            return f"[ERROR: start_line {sl} past end of file (file has {current} lines)]"

        text = "".join(lines_out)
        if len(text) > MAX_READ_BYTES:
            text = text[:MAX_READ_BYTES] + "\n... [truncated at byte cap — narrow line range]"
        header = f"(lines {sl}-{current if lines_out else sl}, path={fp})\n"
        return header + text

    except PermissionError as exc:
        return str(exc)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def read_workspace_attachment(path: str, *, max_bytes: int | None = None) -> tuple[str, bytes] | None:
    """Read a workspace file for outbound delivery (e.g. Discord upload)."""
    cap = max_bytes if max_bytes is not None else settings.attachment_max_bytes
    try:
        fp = _safe_path(path)
        if not fp.is_file():
            return None
        size = fp.stat().st_size
        if size <= 0 or size > cap:
            return None
        return fp.name, fp.read_bytes()
    except Exception:
        return None


def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file, creating parent directories as needed.

    Args:
        path: File path (relative to workspace or absolute).
        content: Text content to write.
        encoding: Text encoding to use when writing.

    Returns:
        Success message or error.
    """
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding=encoding)
        log.info("file_written", path=str(fp), size=len(content))
        return f"Written {len(content)} bytes to {fp}"
    except PermissionError as exc:
        return str(exc)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def list_dir(path: str = ".") -> str:
    """
    List directory contents with file sizes and types.

    Args:
        path: Directory path. Defaults to workspace root.

    Returns:
        Formatted directory listing.
    """
    try:
        dp = _safe_path(path)
        if not dp.exists():
            return f"[ERROR: path not found: {path}]"
        if not dp.is_dir():
            return f"[ERROR: not a directory: {path}]"

        entries = sorted(dp.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return f"(empty directory: {dp})"

        lines = [f"Contents of {dp}:", ""]
        for entry in entries:
            try:
                s = entry.stat()
                kind = "DIR " if entry.is_dir() else "FILE"
                size = f"{s.st_size:>10,} B" if entry.is_file() else "           "
                lines.append(f"  {kind}  {size}  {entry.name}")
            except OSError:
                lines.append(f"  ????             {entry.name}")

        return "\n".join(lines)
    except PermissionError as exc:
        return str(exc)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def delete_file(path: str) -> str:
    """
    Delete a file (not a directory).

    Args:
        path: File path to delete.

    Returns:
        Success message or error.
    """
    try:
        fp = _safe_path(path)
        if not fp.exists():
            return f"[ERROR: file not found: {path}]"
        if fp.is_dir():
            return "[ERROR: use shell 'rm -rf' for directories]"
        fp.unlink()
        return f"Deleted {fp}"
    except PermissionError as exc:
        return str(exc)
    except Exception as exc:
        return f"[ERROR: {exc}]"


def str_replace_file(
    path: str,
    old_str: str,
    new_str: str,
    expected_replacements: int = 1,
) -> str:
    """
    Replace an exact string in a file and write it back.

    Verifies the old string appears exactly expected_replacements times before
    writing, so the agent gets a loud error instead of a silent wrong edit.

    Args:
        path: File path (relative to workspace or absolute).
        old_str: The exact text to find. Include enough surrounding context to
                 make it unique.
        new_str: The replacement text.
        expected_replacements: How many times old_str must appear (default 1).

    Returns:
        Success message or error describing the mismatch.
    """
    try:
        fp = _safe_path(path)
        if not fp.exists():
            return f"[ERROR: file not found: {path}]"
        if not fp.is_file():
            return f"[ERROR: not a file: {path}]"

        content = fp.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_str)
        if count == 0:
            return (
                f"[ERROR: old_str not found in {path}. "
                f"Check for whitespace or encoding differences.]"
            )
        if count != expected_replacements:
            return (
                f"[ERROR: expected {expected_replacements} occurrence(s) of old_str "
                f"but found {count} in {path}. "
                f"Make old_str more specific or set expected_replacements={count}.]"
            )

        updated = content.replace(old_str, new_str, expected_replacements)
        fp.write_text(updated, encoding="utf-8")
        log.info("str_replace", path=str(fp), count=count)
        return f"Replaced {count} occurrence(s) in {fp}"
    except PermissionError as exc:
        return str(exc)
    except Exception as exc:
        return f"[ERROR: {exc}]"


MAX_SEARCH_BYTES = 8 * 1024  # 8 KB — enough for ~200 lines of context


def _rg_base_cmd(
    *,
    context_lines: int,
    file_glob: str,
    fixed_string: bool,
    case_sensitive: bool,
    max_matches_per_file: int | None,
) -> list[str]:
    cmd: list[str] = [
        "rg",
        "--line-number",
        f"--context={context_lines}",
        "--no-heading",
        "--no-messages",
    ]
    if fixed_string:
        cmd.append("--fixed-strings")
    if not case_sensitive:
        cmd.append("--ignore-case")
    else:
        cmd.append("--case-sensitive")
    if max_matches_per_file is not None and max_matches_per_file > 0:
        cmd.append(f"--max-count={int(max_matches_per_file)}")
    if file_glob:
        cmd += ["--glob", file_glob]
    return cmd


def _collect_rg_json_matches(
    stdout: str,
    max_total_matches: int | None,
) -> tuple[list[dict[str, object]], bool]:
    """Parse ripgrep --json lines into match records; truncated if over max_total."""
    out: list[dict[str, object]] = []
    truncated = False
    n = 0
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        n += 1
        if max_total_matches is not None and n > max_total_matches:
            truncated = True
            break
        data = obj.get("data") or {}
        path_obj = data.get("path") or {}
        ptext = path_obj.get("text") or ""
        ln = data.get("line_number")
        ltext = ((data.get("lines") or {}).get("text") or "").rstrip("\n")
        out.append({"path": ptext, "line_number": ln, "line": ltext})
    return out, truncated


def _matches_as_text(matches: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for m in matches:
        p = str(m.get("path", ""))
        ln = m.get("line_number", "")
        lt = str(m.get("line", ""))
        lines.append(f"{p}:{ln}:{lt}")
    return "\n".join(lines)


def search_files(
    pattern: str,
    path: str = ".",
    file_glob: str = "",
    context_lines: int = 2,
    *,
    fixed_string: bool = False,
    case_sensitive: bool = True,
    max_matches_per_file: int | None = None,
    max_total_matches: int | None = None,
    output_mode: str = "text",
) -> str:
    """
    Search files using ripgrep (rg). Respects .gitignore and ignore files by default
    (same as running rg in the repo); files ignored by git are not searched unless
    you narrow path to an unignored subtree.

    Args:
        pattern: Search pattern (regex unless fixed_string=True).
        path: Directory or file to search (relative to workspace or absolute).
        file_glob: File name filter, e.g. '*.py' or '*.{ts,tsx}'.
        context_lines: Lines of context around each match (text mode only; omitted
            when output_mode is json or when max_total_matches forces json pipeline).
        fixed_string: If True, pattern is a literal string (-F).
        case_sensitive: If False, search case-insensitively (-i).
        max_matches_per_file: Ripgrep --max-count per file (optional).
        max_total_matches: Stop after this many matches across the search (uses JSON
            pipeline; context may be omitted in reconstructed text).
        output_mode: 'text' (default, file:line:content) or 'json' (structured array).

    Returns:
        Ripgrep-style text, JSON string, or an explicit no-match / error message.
    """
    try:
        resolved = _safe_path(path)
        if not resolved.exists():
            return f"[ERROR: path not found: {path}]"

        use_json = output_mode.strip().lower() == "json" or (
            max_total_matches is not None and max_total_matches > 0
        )

        if use_json:
            cmd = _rg_base_cmd(
                context_lines=0 if max_total_matches else context_lines,
                file_glob=file_glob,
                fixed_string=fixed_string,
                case_sensitive=case_sensitive,
                max_matches_per_file=max_matches_per_file,
            )
            cmd.append("--json")
            cmd += [pattern, str(resolved)]
        else:
            cmd = _rg_base_cmd(
                context_lines=context_lines,
                file_glob=file_glob,
                fixed_string=fixed_string,
                case_sensitive=case_sensitive,
                max_matches_per_file=max_matches_per_file,
            )
            cmd += [pattern, str(resolved)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        stderr = (result.stderr or "").strip()

        if result.returncode not in (0, 1):
            err_bit = stderr[:300] if stderr else "(no stderr)"
            return (
                f"[ERROR: ripgrep failed (exit {result.returncode}) — not the same as "
                f"'no matches'. Detail: {err_bit}]"
            )

        if use_json:
            matches, truncated = _collect_rg_json_matches(
                result.stdout,
                max_total_matches,
            )
            if not matches:
                return f"(no matches for pattern: {pattern!r})"
            if output_mode.strip().lower() == "json":
                payload: dict[str, object] = {
                    "matches": matches,
                    "truncated": truncated,
                    "gitignore": (
                        "rg respects .gitignore; skipped files do not appear here."
                    ),
                }
                return json.dumps(payload, indent=2)

            text_out = _matches_as_text(matches)
            if truncated:
                text_out += (
                    f"\n... [truncated at max_total_matches={max_total_matches}; "
                    "context omitted in this summary mode]"
                )
            if len(text_out) > MAX_SEARCH_BYTES:
                text_out = (
                    text_out[:MAX_SEARCH_BYTES]
                    + "\n... [truncated at 8KB — refine pattern, path, or file_glob]"
                )
            return text_out

        output = result.stdout
        if not output and result.returncode == 1:
            return f"(no matches for pattern: {pattern!r})"
        if not output and result.returncode == 0:
            return f"(no matches for pattern: {pattern!r})"

        if len(output) > MAX_SEARCH_BYTES:
            output = (
                output[:MAX_SEARCH_BYTES]
                + "\n... [truncated at 8KB — refine your pattern or file_glob]"
            )
        return output

    except FileNotFoundError:
        return "[ERROR: ripgrep (rg) not found — install it with: apt-get install ripgrep]"
    except PermissionError as exc:
        return str(exc)
    except subprocess.TimeoutExpired:
        return "[ERROR: search timed out after 30s — narrow the search path or pattern]"
    except Exception as exc:
        return f"[ERROR: {exc}]"


def apply_patch(path: str, patch: str) -> str:
    """Apply a unified diff patch to a single file."""
    if not patch.strip():
        return "[ERROR: patch is empty]"
    try:
        resolved = _safe_path(path)
    except PermissionError as exc:
        return str(exc)
    if not resolved.exists():
        return f"[ERROR: file not found: {path}]"

    original = resolved.read_text(encoding="utf-8")
    updated, err = _apply_unified_patch(original, patch)
    if err:
        return err
    resolved.write_text(updated, encoding="utf-8")
    log.info("apply_patch", path=str(resolved))
    return f"Patched {resolved} ({len(patch)} chars of diff)."


def _apply_unified_patch(original: str, patch: str) -> tuple[str, str | None]:
    import re

    if "<<<<<<<" in patch and "=======" in patch and ">>>>>>>" in patch:
        return _apply_conflict_markers(original, patch)

    lines = patch.splitlines()
    file_lines = original.splitlines(keepends=True)
    if original and not original.endswith("\n"):
        file_lines = [line + "\n" for line in original.splitlines()]
    out: list[str] = []
    idx = 0
    hunk_header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("@@"):
            i += 1
            continue
        match = hunk_header.match(line)
        if not match:
            return original, f"[ERROR: invalid hunk header: {line}]"
        old_start = int(match.group(1)) - 1
        while idx < old_start:
            if idx >= len(file_lines):
                return original, f"[ERROR: patch starts beyond file end at line {old_start + 1}]"
            out.append(file_lines[idx])
            idx += 1
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            hline = lines[i]
            if hline.startswith(" "):
                if idx >= len(file_lines):
                    return original, "[ERROR: patch extends beyond file end]"
                out.append(file_lines[idx])
                idx += 1
            elif hline.startswith("-"):
                if idx >= len(file_lines):
                    return original, "[ERROR: delete beyond file end]"
                if file_lines[idx].rstrip("\n") != hline[1:].rstrip("\n"):
                    return original, f"[ERROR: delete mismatch at line {idx + 1}]"
                idx += 1
            elif hline.startswith("+"):
                addition = hline[1:]
                if not addition.endswith("\n"):
                    addition += "\n"
                out.append(addition)
            elif hline.startswith("\\"):
                pass
            else:
                return original, f"[ERROR: invalid patch line: {hline}]"
            i += 1
    out.extend(file_lines[idx:])
    if not out and patch.strip():
        return original, "[ERROR: unsupported patch format — use unified diff or conflict markers]"
    if idx == 0 and not any(line.startswith("@@") for line in lines) and "<<<<<<<" not in patch:
        return original, "[ERROR: unsupported patch format — use unified diff or conflict markers]"
    return "".join(out), None


def _apply_conflict_markers(original: str, patch: str) -> tuple[str, str | None]:
    marker_old = "<<<<<<<"
    marker_sep = "======="
    marker_new = ">>>>>>>"
    before, rest = patch.split(marker_old, 1)
    old_part, rest2 = rest.split(marker_sep, 1)
    new_part, after = rest2.split(marker_new, 1)
    needle = old_part.strip("\n")
    if needle not in original:
        return original, "[ERROR: patch old block not found in file]"
    updated = original.replace(needle, new_part.strip("\n"), 1)
    if before or after:
        updated = before + updated + after
    return updated, None
