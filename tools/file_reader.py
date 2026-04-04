# tools/file_reader.py
# ─────────────────────────────────────────────────────────────────
# Fix: BaseTool now imports from 'crewai.tools' not 'crewai_tools'
# ─────────────────────────────────────────────────────────────────

from crewai.tools import BaseTool          # ← FIXED import
from pathlib import Path


class FileReaderTool(BaseTool):
    name: str = "File Reader"
    description: str = (
        "Read content from a local file (PDF, TXT, DOCX). "
        "Input: file path as a string. "
        "Output: extracted text content of the file."
    )

    def _run(self, file_path: str) -> str:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File not found at {file_path}"

        if path.suffix in [".txt", ".md"]:
            return path.read_text(encoding="utf-8")

        elif path.suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() for page in reader.pages)
            return text

        elif path.suffix == ".docx":
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)

        else:
            return f"Unsupported file type: {path.suffix}"