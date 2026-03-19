"""Attachment ingestion for Discord-originated tasks."""

from __future__ import annotations

import csv
import io
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import openpyxl
import structlog
import xlrd
from PIL import Image, UnidentifiedImageError
from pydantic_ai.messages import BinaryContent
from pypdf import PdfReader

log = structlog.get_logger()

_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".xml",
    ".sql",
}
_CSV_EXTENSIONS = {".csv", ".tsv"}
_SPREADSHEET_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_FILENAME_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class AttachmentBundle:
    prompt_text: str = ""
    metadata: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_attachments(self) -> bool:
        return bool(self.metadata)


def _safe_filename(name: str, *, fallback: str) -> str:
    cleaned = _FILENAME_CLEAN_RE.sub("-", name.strip()) or fallback
    return cleaned[:120]


def _truncate(text: str, char_cap: int) -> str:
    stripped = text.strip()
    if len(stripped) <= char_cap:
        return stripped
    return stripped[:char_cap] + "\n... [truncated]"


def _guess_content_type(filename: str, declared: str | None) -> str:
    if declared:
        return declared
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _decode_text(data: bytes, *, char_cap: int) -> str:
    return _truncate(data.decode("utf-8", errors="replace"), char_cap)


def _preview_csv(data: bytes, *, delimiter: str, char_cap: int) -> str:
    text = data.decode("utf-8", errors="replace")
    rows: list[str] = []
    for i, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter)):
        if i >= 20:
            rows.append("... [additional rows omitted]")
            break
        clipped = [cell[:80] + ("..." if len(cell) > 80 else "") for cell in row[:10]]
        rows.append(" | ".join(clipped))
    return _truncate("\n".join(rows), char_cap)


def _preview_xlsx(data: bytes, *, char_cap: int) -> str:
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in workbook.worksheets[:3]:
        lines.append(f"[Sheet] {sheet.title}")
        for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            if row_idx > 15:
                lines.append("... [additional rows omitted]")
                break
            cells = ["" if value is None else str(value)[:80] for value in row[:10]]
            lines.append(" | ".join(cells))
    return _truncate("\n".join(lines), char_cap)


def _preview_xls(data: bytes, *, char_cap: int) -> str:
    workbook = xlrd.open_workbook(file_contents=data)
    lines: list[str] = []
    for sheet in workbook.sheets()[:3]:
        lines.append(f"[Sheet] {sheet.name}")
        for row_idx in range(min(sheet.nrows, 15)):
            cells = [str(sheet.cell_value(row_idx, col_idx))[:80] for col_idx in range(min(sheet.ncols, 10))]
            lines.append(" | ".join(cells))
        if sheet.nrows > 15:
            lines.append("... [additional rows omitted]")
    return _truncate("\n".join(lines), char_cap)


def _preview_pdf(data: bytes, *, char_cap: int) -> tuple[str, dict[str, Any]]:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages[:5]:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages.append(extracted.strip())
    preview = _truncate("\n\n".join(pages) or "(no extractable text found)", char_cap)
    return preview, {"page_count": len(reader.pages)}


def _image_details(data: bytes) -> tuple[str, dict[str, Any]]:
    with Image.open(io.BytesIO(data)) as image:
        details = {
            "format": image.format or "unknown",
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }
    summary = (
        f"Image metadata: {details['format']} {details['width']}x{details['height']} "
        f"({details['mode']})."
    )
    return summary, details


def render_attachment_context(attachments: Sequence[dict[str, Any]]) -> str:
    if not attachments:
        return ""
    lines = ["## Attachments"]
    for attachment in attachments:
        filename = str(attachment.get("filename", "attachment"))
        content_type = str(attachment.get("content_type", "unknown"))
        size_bytes = int(attachment.get("size_bytes", 0) or 0)
        saved_path = str(attachment.get("saved_path", ""))
        summary = str(attachment.get("summary", "")).strip()
        lines.append(f"### {filename}")
        lines.append(f"- type: {content_type}")
        lines.append(f"- size: {size_bytes} bytes")
        if saved_path:
            lines.append(f"- saved path: {saved_path}")
        if summary:
            lines.append(summary)
    return "\n".join(lines)


async def ingest_discord_attachments(
    attachments: Sequence[Any],
    *,
    root: Path,
    storage_key: str,
    max_bytes: int,
    text_char_cap: int,
) -> AttachmentBundle:
    if not attachments:
        return AttachmentBundle()

    target_dir = root / _safe_filename(storage_key, fallback="message")
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, Any]] = []
    for index, attachment in enumerate(attachments, start=1):
        raw_name = str(getattr(attachment, "filename", "") or f"attachment-{index}")
        safe_name = _safe_filename(raw_name, fallback=f"attachment-{index}")
        content_type = _guess_content_type(safe_name, getattr(attachment, "content_type", None))
        try:
            data = await attachment.read()
        except Exception as exc:
            log.warning("attachment_read_failed", filename=raw_name, error=str(exc))
            metadata.append(
                {
                    "filename": safe_name,
                    "content_type": content_type,
                    "size_bytes": int(getattr(attachment, "size", 0) or 0),
                    "saved_path": "",
                    "summary": f"Failed to read attachment: {exc}",
                    "inline_part": None,
                }
            )
            continue

        size_bytes = len(data)
        saved_path = target_dir / safe_name
        saved_path.write_bytes(data)

        suffix = Path(safe_name).suffix.lower()
        summary = ""
        inline_part: dict[str, Any] | None = None

        try:
            if size_bytes > max_bytes:
                summary = (
                    f"Attachment saved but inline parsing skipped because it exceeds the "
                    f"configured size cap ({size_bytes} > {max_bytes} bytes)."
                )
            elif suffix in _CSV_EXTENSIONS or content_type in {"text/csv", "text/tab-separated-values"}:
                delimiter = "\t" if suffix == ".tsv" or content_type == "text/tab-separated-values" else ","
                summary = "CSV preview:\n" + _preview_csv(data, delimiter=delimiter, char_cap=text_char_cap)
            elif suffix in _SPREADSHEET_EXTENSIONS:
                preview = _preview_xls(data, char_cap=text_char_cap) if suffix == ".xls" else _preview_xlsx(
                    data,
                    char_cap=text_char_cap,
                )
                summary = "Spreadsheet preview:\n" + preview
            elif suffix == ".pdf" or content_type == "application/pdf":
                preview, extra = _preview_pdf(data, char_cap=text_char_cap)
                summary = f"PDF preview ({extra['page_count']} pages):\n{preview}"
                inline_part = {
                    "path": str(saved_path),
                    "media_type": "application/pdf",
                    "identifier": safe_name,
                    "vendor_metadata": None,
                }
            elif suffix in _IMAGE_EXTENSIONS or content_type.startswith("image/"):
                details_summary, details = _image_details(data)
                summary = details_summary
                inline_part = {
                    "path": str(saved_path),
                    "media_type": content_type,
                    "identifier": safe_name,
                    "vendor_metadata": {"detail": "high"},
                    "details": details,
                }
            elif suffix in _TEXT_EXTENSIONS or content_type.startswith("text/"):
                summary = "Text preview:\n" + _decode_text(data, char_cap=text_char_cap)
            else:
                summary = "Saved attachment for later inspection. No inline parser is configured for this file type."
        except (UnicodeDecodeError, UnidentifiedImageError, ValueError, xlrd.XLRDError) as exc:
            summary = f"Attachment saved, but preview extraction failed: {exc}"
        except Exception as exc:
            log.warning("attachment_parse_failed", filename=safe_name, error=str(exc))
            summary = f"Attachment saved, but preview extraction failed: {exc}"

        metadata.append(
            {
                "filename": safe_name,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "saved_path": str(saved_path),
                "summary": summary,
                "inline_part": inline_part,
            }
        )

    return AttachmentBundle(prompt_text=render_attachment_context(metadata), metadata=metadata)


def inline_prompt_parts_from_metadata(attachments: Sequence[dict[str, Any]]) -> list[BinaryContent]:
    parts: list[BinaryContent] = []
    for attachment in attachments:
        inline_part = attachment.get("inline_part")
        if not isinstance(inline_part, dict):
            continue
        path = inline_part.get("path")
        media_type = inline_part.get("media_type")
        if not isinstance(path, str) or not isinstance(media_type, str):
            continue
        try:
            data = Path(path).read_bytes()
        except OSError:
            continue
        parts.append(
            BinaryContent(
                data=data,
                media_type=media_type,
                identifier=str(inline_part.get("identifier") or Path(path).name),
                vendor_metadata=inline_part.get("vendor_metadata"),
            )
        )
    return parts
