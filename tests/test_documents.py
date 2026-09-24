"""事前資料（PDF・Word・Excel・PowerPoint）から本文を取り出すところ。

新しい依存は入れていない（PDF は macOS の PDFKit、Office は zip + XML）。
ここでは**手で組み立てた最小のファイル**を読ませる。実物の Office 文書は名前空間が付くが、
こちらは名前空間を無視して読むので同じ経路を通る。
"""

from __future__ import annotations

import zipfile
from io import BytesIO

import pytest

from src.text.documents import UnreadableDocument, extract_text

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _zip(files: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _pdf(lines: list[str]) -> bytes:
    """1 ページに Helvetica で数行だけ書いた PDF（圧縮なし）。"""
    body = ["BT", "/F1 18 Tf", "72 720 Td", "20 TL"]
    for index, line in enumerate(lines):
        body.append(f"({line}) Tj" if index == 0 else f"T* ({line}) Tj")
    body.append("ET")
    stream = "\n".join(body).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_pdf_の本文を取り出す():
    text = extract_text("議題.pdf", _pdf(["Agenda", "1. Quote review", "2. Next site"]))
    assert "Agenda" in text and "Quote review" in text


def test_文字が入っていない_pdf_は断る():
    """スキャンしただけの PDF。黙って空を読み込むと「付けたのに効かない」になる。"""
    with pytest.raises(UnreadableDocument, match="スキャン"):
        extract_text("スキャン.pdf", _pdf([]))


def test_word_は段落ごとに行にする():
    document = f"""<?xml version="1.0"?>
    <w:document xmlns:w="{W}"><w:body>
      <w:p><w:r><w:t>見積の確認</w:t></w:r><w:r><w:t>（9/20 まで）</w:t></w:r></w:p>
      <w:p><w:r><w:t>次期サイトの範囲</w:t></w:r></w:p>
      <w:p><w:r><w:t>   </w:t></w:r></w:p>
    </w:body></w:document>"""
    text = extract_text("議事メモ.docx", _zip({"word/document.xml": document}))
    assert text.splitlines() == ["見積の確認（9/20 まで）", "次期サイトの範囲"]


def test_powerpoint_はスライドごとにまとめてノートも拾う():
    slide = f"""<?xml version="1.0"?><p:sld xmlns:p="{P}" xmlns:a="{A}"><a:t>次期サイトの提案</a:t>
      <a:t>費用は 300 万円</a:t></p:sld>"""
    notes = f"""<?xml version="1.0"?><p:notes xmlns:p="{P}" xmlns:a="{A}"><a:t>予算は未確定と伝える</a:t></p:notes>"""
    text = extract_text("提案.pptx", _zip({"ppt/slides/slide1.xml": slide,
                                          "ppt/notesSlides/notesSlide1.xml": notes}))
    assert "#### スライド 1" in text
    assert "費用は 300 万円" in text
    assert "（ノート）予算は未確定と伝える" in text     # ノートは本文より濃いことがある


def test_excel_はシートごとにタブ区切りにする():
    workbook = f"""<?xml version="1.0"?><workbook xmlns="{S}"><sheets>
      <sheet name="見積" sheetId="1"/></sheets></workbook>"""
    shared = f"""<?xml version="1.0"?><sst xmlns="{S}"><si><t>項目</t></si><si><t>金額</t></si>
      <si><t>制作費</t></si></sst>"""
    sheet = f"""<?xml version="1.0"?><worksheet xmlns="{S}"><sheetData>
      <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
      <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>1200000</v></c></row>
      <row r="3"/>
    </sheetData></worksheet>"""
    text = extract_text("見積.xlsx", _zip({"xl/workbook.xml": workbook,
                                          "xl/sharedStrings.xml": shared,
                                          "xl/worksheets/sheet1.xml": sheet}))
    assert text.splitlines() == ["#### 見積", "項目\t金額", "制作費\t1200000"]


def test_長い表は途中で打ち切る():
    rows = "".join(f'<row r="{index}"><c r="A{index}"><v>{index}</v></c></row>'
                   for index in range(1, 900))
    sheet = f"""<?xml version="1.0"?><worksheet xmlns="{S}"><sheetData>{rows}</sheetData></worksheet>"""
    text = extract_text("長い表.xlsx", _zip({"xl/worksheets/sheet1.xml": sheet}))
    assert "以降は省略" in text
    assert len(text.splitlines()) < 520


def test_知らない種類は断る():
    with pytest.raises(UnreadableDocument, match="読めません"):
        extract_text("写真.heic", b"\x00\x01")


def test_壊れたファイルは断る():
    with pytest.raises(UnreadableDocument):
        extract_text("壊れた.docx", "これは zip ではない".encode())


def test_文字のファイルはそのまま読む():
    assert extract_text("議題.md", "## 議題\n- 見積".encode()) == "## 議題\n- 見積"
