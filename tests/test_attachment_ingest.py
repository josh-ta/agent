from __future__ import annotations

import pytest

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
