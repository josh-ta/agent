from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.attachment_ingest as attachment_ingest
from agent.attachment_ingest import ingest_discord_attachments, inline_prompt_parts_from_metadata
from tests.conftest import FakeDiscordAttachment


@pytest.mark.asyncio
async def test_ingest_discord_attachments_supports_csv_pdf_and_images(tmp_path) -> None:
    attachments = [
        FakeDiscordAttachment(
            filename="report.csv",
            data=b"name,value\nfoo,1\nbar,2\n",
            content_type="text/csv",
        ),
        FakeDiscordAttachment(
            filename="doc.pdf",
            data=(
                b"%PDF-1.4\n"
                b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
                b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj\n"
                b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 72 120 Td (hello pdf) Tj ET\nendstream endobj\n"
                b"xref\n0 5\n0000000000 65535 f \n"
                b"0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000207 00000 n \n"
                b"trailer<</Root 1 0 R/Size 5>>\nstartxref\n301\n%%EOF\n"
            ),
            content_type="application/pdf",
        ),
        FakeDiscordAttachment(
            filename="image.png",
            data=(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
                b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
            ),
            content_type="image/png",
        ),
    ]

    bundle = await ingest_discord_attachments(
        attachments,
        root=tmp_path,
        storage_key="discord-101-1",
        max_bytes=1_000_000,
        text_char_cap=4_000,
    )

    assert len(bundle.metadata) == 3
    assert "report.csv" in bundle.prompt_text
    assert "doc.pdf" in bundle.prompt_text
    assert "image.png" in bundle.prompt_text

    prompt_parts = inline_prompt_parts_from_metadata(bundle.metadata)
    assert len(prompt_parts) == 2
    assert {part.media_type for part in prompt_parts} == {"application/pdf", "image/png"}


def test_attachment_helper_functions_cover_common_branches(tmp_path) -> None:
    assert attachment_ingest._safe_filename("  bad name?.txt  ", fallback="fallback") == "bad-name-.txt"
    assert attachment_ingest._safe_filename("   ", fallback="fallback") == "fallback"
    assert attachment_ingest._truncate("hello", 10) == "hello"
    assert attachment_ingest._truncate("  hello world  ", 5).endswith("[truncated]")
    assert attachment_ingest._guess_content_type("note.txt", "text/plain") == "text/plain"
    assert attachment_ingest._guess_content_type("note.txt", None) == "text/plain"
    assert attachment_ingest._guess_content_type("blob.unknownext", None) == "application/octet-stream"
    assert attachment_ingest._decode_text(b"hello\xff", char_cap=20) == "hello�"

    csv_preview = attachment_ingest._preview_csv(
        "\n".join(f"a,b,{i}" for i in range(25)).encode(),
        delimiter=",",
        char_cap=10_000,
    )
    assert "... [additional rows omitted]" in csv_preview

    saved = tmp_path / "file.txt"
    saved.write_text("x", encoding="utf-8")
    rendered = attachment_ingest.render_attachment_context(
        [
            {
                "filename": "file.txt",
                "content_type": "text/plain",
                "size_bytes": 1,
                "saved_path": str(saved),
                "summary": "Text preview",
            }
        ]
    )
    assert "## Attachments" in rendered
    assert str(saved) in rendered
    assert attachment_ingest.render_attachment_context([]) == ""


def test_attachment_preview_helpers_cover_xlsx_xls_pdf_and_image(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Sheet:
        def __init__(self, title: str) -> None:
            self.title = title

        def iter_rows(self, values_only=True):
            del values_only
            yield ("a", "b")
            yield ("c", None)

    class _Workbook:
        worksheets = [_Sheet("One")]

    monkeypatch.setattr(attachment_ingest.openpyxl, "load_workbook", lambda *args, **kwargs: _Workbook())
    xlsx_preview = attachment_ingest._preview_xlsx(b"data", char_cap=500)
    assert "[Sheet] One" in xlsx_preview

    class _XlsSheet:
        name = "Legacy"
        nrows = 2
        ncols = 2

        def cell_value(self, row_idx: int, col_idx: int) -> str:
            return f"{row_idx}:{col_idx}"

    class _XlsBook:
        def sheets(self):
            return [_XlsSheet()]

    monkeypatch.setattr(attachment_ingest.xlrd, "open_workbook", lambda **kwargs: _XlsBook())
    xls_preview = attachment_ingest._preview_xls(b"data", char_cap=500)
    assert "[Sheet] Legacy" in xls_preview

    class _PdfPage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _PdfReader:
        def __init__(self, *_args, **_kwargs) -> None:
            self.pages = [_PdfPage("first page"), _PdfPage("")]

    monkeypatch.setattr(attachment_ingest, "PdfReader", _PdfReader)
    pdf_preview, extra = attachment_ingest._preview_pdf(b"%PDF", char_cap=500)
    assert "first page" in pdf_preview
    assert extra["page_count"] == 2

    class _Image:
        format = "PNG"
        width = 10
        height = 20
        mode = "RGB"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(attachment_ingest.Image, "open", lambda *_args, **_kwargs: _Image())
    summary, details = attachment_ingest._image_details(b"img")
    assert "10x20" in summary
    assert details["mode"] == "RGB"


@pytest.mark.asyncio
async def test_ingest_discord_attachments_covers_failures_and_misc_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _UnreadableAttachment:
        filename = "broken.bin"
        content_type = "application/octet-stream"
        size = 12

        async def read(self) -> bytes:
            raise RuntimeError("boom")

    huge = FakeDiscordAttachment(filename="huge.bin", data=b"x" * 20, content_type="application/octet-stream")
    text = FakeDiscordAttachment(filename="note.md", data=b"# hello", content_type="text/markdown")
    unknown = FakeDiscordAttachment(filename="archive.bin", data=b"abc", content_type="application/octet-stream")

    bundle = await ingest_discord_attachments(
        [_UnreadableAttachment(), huge, text, unknown],
        root=tmp_path,
        storage_key="discord-1-1",
        max_bytes=10,
        text_char_cap=500,
    )

    assert "Failed to read attachment" in bundle.metadata[0]["summary"]
    assert "exceeds the configured size cap" in bundle.metadata[1]["summary"]
    assert "Text preview" in bundle.metadata[2]["summary"]
    assert "No inline parser is configured" in bundle.metadata[3]["summary"]

    broken_image = FakeDiscordAttachment(filename="broken.png", data=b"img", content_type="image/png")
    monkeypatch.setattr(attachment_ingest, "_image_details", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad image")))
    image_bundle = await ingest_discord_attachments(
        [broken_image],
        root=tmp_path,
        storage_key="discord-1-2",
        max_bytes=500,
        text_char_cap=500,
    )
    assert "preview extraction failed" in image_bundle.metadata[0]["summary"]

    broken_pdf = FakeDiscordAttachment(filename="broken.pdf", data=b"%PDF", content_type="application/pdf")
    monkeypatch.setattr(attachment_ingest, "_preview_pdf", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad pdf")))
    pdf_bundle = await ingest_discord_attachments(
        [broken_pdf],
        root=tmp_path,
        storage_key="discord-1-3",
        max_bytes=500,
        text_char_cap=500,
    )
    assert "preview extraction failed" in pdf_bundle.metadata[0]["summary"]


def test_inline_prompt_parts_from_metadata_skips_invalid_entries_and_missing_files(tmp_path) -> None:
    existing = tmp_path / "doc.pdf"
    existing.write_bytes(b"pdf")
    parts = inline_prompt_parts_from_metadata(
        [
            {"inline_part": None},
            {"inline_part": {"path": 123, "media_type": "application/pdf"}},
            {"inline_part": {"path": str(tmp_path / "missing.pdf"), "media_type": "application/pdf"}},
            {
                "inline_part": {
                    "path": str(existing),
                    "media_type": "application/pdf",
                    "identifier": "doc.pdf",
                    "vendor_metadata": {"detail": "high"},
                }
            },
        ]
    )

    assert len(parts) == 1
    assert parts[0].identifier == "doc.pdf"
