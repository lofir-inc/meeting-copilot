"""会議の事前資料（PDF・Word・Excel・PowerPoint）から本文を取り出す。

**新しい依存は要らない**（2026-09-14 に確かめた）。
  - PDF … macOS の PDFKit（`Quartz.PDFDocument`）。pyobjc は ScreenCaptureKit で既に入っている。
    入っていない環境のために `pypdf` があればそちらも使う
  - Word / Excel / PowerPoint … 中身は **zip + XML** なので、`zipfile` と `xml.etree` だけで読める

取り出すのは**文字だけ**。図・写真・レイアウトは落ちる。スキャンしただけの PDF（文字が入って
いない）は空になるので、その場合は「文字が入っていない」と言って断る — 黙って空の資料を
読み込むと、**資料を付けたのに効かない**という分かりにくい状態になる（画面から足せるので、
そのときは本文を貼り付けてもらう）。
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".vtt"}
"""そのまま文字として読めるもの。"""

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx"}
"""中身を取り出してから読むもの。"""

MAX_SHEET_ROWS = 500
"""1 シートから読む行数の上限。表は際限なく長くなるので、頭から読んで打ち切る。"""


class UnreadableDocument(Exception):
    """読めなかった（対応していない種類・文字が入っていない・壊れている）。"""


def extract_text(name: str, data: bytes) -> str:
    """ファイル名と中身から本文を取り出す。読めなければ `UnreadableDocument`。"""
    suffix = Path(str(name)).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        return _from_pdf(data)
    if suffix == ".docx":
        return _from_docx(data)
    if suffix == ".pptx":
        return _from_pptx(data)
    if suffix == ".xlsx":
        return _from_xlsx(data)
    raise UnreadableDocument(
        f"{suffix or 'この種類'} は読めません（"
        f"{'・'.join(sorted(TEXT_SUFFIXES | DOCUMENT_SUFFIXES))}）"
    )


# --------------------------------------------------------------------- PDF

def _from_pdf(data: bytes) -> str:
    """PDF の文字を取り出す。macOS の PDFKit → pypdf の順で試す。"""
    text = _pdf_with_pdfkit(data)
    if text is None:
        text = _pdf_with_pypdf(data)
    if text is None:
        raise UnreadableDocument("PDF を読む手段がありません（macOS の PDFKit も pypdf も使えません）")
    if not text.strip():
        raise UnreadableDocument(
            "文字が入っていない PDF です（スキャンした画像？）。本文を貼り付けて足してください"
        )
    return text


def _pdf_with_pdfkit(data: bytes) -> str | None:
    """macOS の PDFKit で読む。pyobjc は ScreenCaptureKit で既に入っている。"""
    try:
        import Quartz
        from Foundation import NSData
    except ImportError:
        return None
    blob = NSData.dataWithBytes_length_(data, len(data))
    document = Quartz.PDFDocument.alloc().initWithData_(blob)
    if document is None:
        raise UnreadableDocument("PDF として読めませんでした（壊れている？）")
    pages = [document.pageAtIndex_(index).string() or "" for index in range(document.pageCount())]
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _pdf_with_pypdf(data: bytes) -> str | None:
    """pypdf があれば使う（macOS 以外のための保険）。"""
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page)


# ------------------------------------------------------- Office（zip + XML）

def _open(data: bytes) -> zipfile.ZipFile:
    from io import BytesIO

    try:
        return zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile as error:
        raise UnreadableDocument("中身を開けませんでした（壊れている？）") from error


def _local(tag: str) -> str:
    """`{名前空間}w:t` のような修飾を外す。"""
    return tag.rsplit("}", 1)[-1]


def _texts(element, tag: str) -> list[str]:
    """指定タグ（名前空間なし）の文字を、出てくる順に集める。"""
    return [node.text or "" for node in element.iter() if _local(node.tag) == tag]


def _from_docx(data: bytes) -> str:
    """Word。段落（w:p）ごとに 1 行にする。表のセルも段落として拾える。"""
    with _open(data) as archive:
        if "word/document.xml" not in archive.namelist():
            raise UnreadableDocument("Word の本文が見つかりません")
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    lines: list[str] = []
    for node in root.iter():
        if _local(node.tag) != "p":
            continue
        line = "".join(_texts(node, "t")).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
    if not text.strip():
        raise UnreadableDocument("文字が入っていない Word です")
    return text


def _from_pptx(data: bytes) -> str:
    """PowerPoint。スライドごとに見出しを付けて、図形と発表者ノートの文字を並べる。"""
    with _open(data) as archive:
        slides = sorted(
            (name for name in archive.namelist()
             if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
            key=_slide_number,
        )
        if not slides:
            raise UnreadableDocument("スライドが見つかりません")
        blocks: list[str] = []
        for slide in slides:
            number = _slide_number(slide)
            root = ElementTree.fromstring(archive.read(slide))
            lines = [value.strip() for value in _texts(root, "t") if value and value.strip()]
            notes_name = f"ppt/notesSlides/notesSlide{number}.xml"
            if notes_name in archive.namelist():
                notes = ElementTree.fromstring(archive.read(notes_name))
                # 発表者ノートには「何を話すつもりか」が書いてある。会議の材料としては本文より濃い
                note_lines = [value.strip() for value in _texts(notes, "t") if value and value.strip()]
                if note_lines:
                    lines += ["（ノート）" + " ".join(note_lines)]
            if lines:
                blocks.append(f"#### スライド {number}\n" + "\n".join(lines))
    if not blocks:
        raise UnreadableDocument("文字が入っていない PowerPoint です（画像だけ？）")
    return "\n\n".join(blocks)


def _slide_number(name: str) -> int:
    found = re.search(r"(\d+)\.xml$", name)
    return int(found.group(1)) if found else 0


def _from_xlsx(data: bytes) -> str:
    """Excel。シートごとに、値の入った行をタブ区切りで並べる（数式は計算済みの値を読む）。"""
    with _open(data) as archive:
        names = archive.namelist()
        shared = _shared_strings(archive) if "xl/sharedStrings.xml" in names else []
        sheets = sorted((name for name in names
                         if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)), key=_slide_number)
        if not sheets:
            raise UnreadableDocument("シートが見つかりません")
        titles = _sheet_titles(archive) if "xl/workbook.xml" in names else []
        blocks: list[str] = []
        for index, sheet in enumerate(sheets):
            title = titles[index] if index < len(titles) else Path(sheet).stem
            rows = _sheet_rows(ElementTree.fromstring(archive.read(sheet)), shared)
            if rows:
                blocks.append(f"#### {title}\n" + "\n".join(rows))
    if not blocks:
        raise UnreadableDocument("文字が入っていない Excel です")
    return "\n\n".join(blocks)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for node in root.iter():
        if _local(node.tag) == "si":
            values.append("".join(_texts(node, "t")))
    return values


def _sheet_titles(archive: zipfile.ZipFile) -> list[str]:
    root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    return [node.attrib.get("name", "") for node in root.iter() if _local(node.tag) == "sheet"]


def _sheet_rows(root, shared: list[str]) -> list[str]:
    """行をタブ区切りの文字列にする。空の行は落とす。"""
    rows: list[str] = []
    for row in (node for node in root.iter() if _local(node.tag) == "row"):
        cells: list[str] = []
        for cell in (node for node in row.iter() if _local(node.tag) == "c"):
            cells.append(_cell_value(cell, shared))
        line = "\t".join(cells).strip()
        if line:
            rows.append(line)
        if len(rows) >= MAX_SHEET_ROWS:
            rows.append(f"（以降は省略。先頭 {MAX_SHEET_ROWS} 行まで読みました）")
            break
    return rows


def _cell_value(cell, shared: list[str]) -> str:
    kind = cell.attrib.get("t", "")
    if kind == "s":                                  # 共有文字列（番号で引く）
        raw = "".join(_texts(cell, "v")).strip()
        index = int(raw) if raw.isdigit() else -1
        return shared[index] if 0 <= index < len(shared) else ""
    if kind == "inlineStr":
        return "".join(_texts(cell, "t")).strip()
    # 数値・日付・数式は、保存されている計算済みの値をそのまま読む
    return "".join(_texts(cell, "v")).strip()
