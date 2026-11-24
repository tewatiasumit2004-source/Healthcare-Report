import io
import os
import re
from pathlib import Path

import fitz
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "20"))
MAX_IMAGE_PIXELS = 20_000_000
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ABNORMAL_MARKER = re.compile(
    r"(?i:\bhigh\b|\blow\b|\babnormal\b|\bcritical\b)|\s(?:H|L|A|C)$"
)

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="Healthcare Report Explainer",
    description=(
        "Extracts text from reports locally and identifies results explicitly "
        "flagged by the laboratory. It does not diagnose."
    ),
    version="1.0.0",
)


class AbnormalReportRequest(BaseModel):
    filename: str = Field(max_length=255)
    summary: str = Field(max_length=2000)
    flagged_lines: list[str] = Field(max_length=100)
    notice: str = Field(max_length=2000)


def _ocr_image(image: Image.Image) -> str:
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image dimensions exceed the allowed limit.")
    try:
        return pytesseract.image_to_string(image, lang=OCR_LANGUAGE).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="OCR is unavailable. Ensure Tesseract is installed in the deployment image.",
        ) from exc
    except pytesseract.TesseractError as exc:
        raise HTTPException(status_code=422, detail="The report could not be read with OCR.") from exc


def _extract_image(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the allowed limit.")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            return _ocr_image(image.convert("RGB"))
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="The image file is invalid or could not be read.") from exc


def _extract_pdf(data: bytes) -> str:
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.is_encrypted:
                raise HTTPException(status_code=422, detail="Password-protected PDFs are not supported.")
            if len(document) == 0:
                raise HTTPException(status_code=422, detail="The PDF contains no pages.")
            if len(document) > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDFs may contain at most {MAX_PDF_PAGES} pages.",
                )

            page_text = []
            for page in document:
                text = page.get_text().strip()
                if len(text) < 20:
                    scale = 2
                    if page.rect.width * scale * page.rect.height * scale > MAX_IMAGE_PIXELS:
                        raise HTTPException(
                            status_code=413,
                            detail="A PDF page exceeds the allowed image dimensions.",
                        )
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    text = _ocr_image(image)
                if text:
                    page_text.append(text)
            return "\n\n".join(page_text)
    except HTTPException:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="The PDF is invalid or could not be read.") from exc


def _flagged_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and ABNORMAL_MARKER.search(line.strip())
    ]


def _pdf_safe_text(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text).encode(
        "cp1252", "replace"
    ).decode("cp1252")


def _wrap_pdf_text(text: str, width: float, font_size: float) -> list[str]:
    words = _pdf_safe_text(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if fitz.get_text_length(candidate, fontname="helv", fontsize=font_size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _create_abnormal_report_pdf(report: AbnormalReportRequest) -> bytes:
    document = fitz.open()
    page = document.new_page()
    margin = 52
    content_width = page.rect.width - 2 * margin
    y = 52

    def next_page() -> None:
        nonlocal page, y
        page = document.new_page()
        y = 52
        page.insert_text(
            (margin, y + 20),
            "ABNORMAL VALUES REPORT (CONTINUED)",
            fontname="helv",
            fontsize=11,
            color=(0.35, 0.39, 0.38),
        )
        y += 40

    def add_paragraph(text: str, fontsize: float, color: tuple[float, float, float]) -> None:
        nonlocal y
        lines = _wrap_pdf_text(text, content_width, fontsize)
        line_height = fontsize * 1.5
        if y + len(lines) * line_height > page.rect.height - margin:
            next_page()
        for line in lines:
            page.insert_text(
                (margin, y + fontsize),
                line,
                fontname="helv",
                fontsize=fontsize,
                color=color,
            )
            y += line_height
        y += 7

    page.draw_rect(
        fitz.Rect(0, 0, page.rect.width, 9),
        color=(0.78, 0.16, 0.16),
        fill=(0.78, 0.16, 0.16),
    )
    add_paragraph("Abnormal Values Report", 21, (0.13, 0.19, 0.18))
    source_filename = Path(report.filename.replace("\\", "/")).name
    add_paragraph(f"Source report: {source_filename}", 10, (0.35, 0.39, 0.38))
    page.draw_line(
        fitz.Point(margin, y),
        fitz.Point(page.rect.width - margin, y),
        color=(0.85, 0.88, 0.87),
        width=1,
    )
    y += 15
    add_paragraph("SUMMARY", 10, (0.42, 0.13, 0.13))
    add_paragraph(report.summary, 11, (0.16, 0.2, 0.19))
    add_paragraph("LAB-MARKED VALUES", 10, (0.42, 0.13, 0.13))

    if report.flagged_lines:
        for flagged_line in report.flagged_lines:
            lines = _wrap_pdf_text(flagged_line, content_width - 24, 11)
            box_height = len(lines) * 17 + 16
            if y + box_height > page.rect.height - margin:
                next_page()
            box = fitz.Rect(margin, y, page.rect.width - margin, y + box_height)
            page.draw_rect(box, color=(0.78, 0.16, 0.16), fill=(1.0, 0.94, 0.94), width=1)
            for index, line in enumerate(lines):
                page.insert_text(
                    (margin + 12, y + 19 + index * 17),
                    line,
                    fontname="helv",
                    fontsize=11,
                    color=(0.56, 0.1, 0.1),
                )
            y += box_height + 8
    else:
        add_paragraph(
            "No high, low, abnormal, or critical markers were detected in the extracted text.",
            11,
            (0.35, 0.39, 0.38),
        )

    if y + 45 > page.rect.height - margin:
        next_page()
    page.draw_line(
        fitz.Point(margin, y + 4),
        fitz.Point(page.rect.width - margin, y + 4),
        color=(0.85, 0.88, 0.87),
        width=1,
    )
    y += 13
    add_paragraph(report.notice, 9, (0.35, 0.39, 0.38))
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/reports/abnormal-values.pdf")
async def download_abnormal_report(report: AbnormalReportRequest) -> StreamingResponse:
    if len(report.filename) == 0 or any(len(line) > 1000 for line in report.flagged_lines):
        raise HTTPException(status_code=422, detail="The report contains invalid or oversized fields.")

    pdf_bytes = _create_abnormal_report_pdf(report)
    source_name = Path(report.filename.replace("\\", "/")).stem
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", source_name).strip("_") or "report"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}-abnormal-values.pdf"'},
    )


@app.post("/api/analyze")
async def analyze_report(file: UploadFile = File(...)) -> dict[str, object]:
    extension = Path(file.filename or "").suffix.lower()
    if extension != ".pdf" and extension not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a PDF or PNG, JPG, JPEG, TIF, or TIFF image.",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"The file exceeds the {limit_mb:g} MB upload limit.",
        )
    if not data:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    if extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail="The file does not appear to be a valid PDF.")
        extracted_text = _extract_pdf(data)
    else:
        extracted_text = _extract_image(data)

    flagged_lines = _flagged_lines(extracted_text)
    if extracted_text:
        summary = (
            f"Text was extracted from the report. "
            f"{len(flagged_lines)} line(s) contain a high, low, abnormal, or critical "
            "marker printed on the report."
        )
    else:
        summary = "No readable text was found. Try a clearer image or a text-based PDF."

    return {
        "filename": Path((file.filename or "report").replace("\\", "/")).name,
        "summary": summary,
        "flagged_lines": flagged_lines,
        "extracted_text": extracted_text,
        "notice": (
            "This educational tool may make OCR mistakes and does not provide a diagnosis. "
            "A flagged result is not necessarily a sign of disease. Review results with a "
            "qualified healthcare professional. Files are not retained after the request."
        ),
    }
