"""
Day 4 script: Extract text and key numbers from a pitch deck PDF.

Tries normal text extraction first (pdfplumber). If a page has no
extractable text, or very little (e.g. just a title, meaning the rest of
the slide is an image), falls back to OCR (Tesseract via pytesseract) --
this converts the page to an image and "reads" the text out of the picture.

Run: python src/pdf_parser.py
"""

import re
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# This PDF has one very tall page (all slides stacked into one page), so
# the rendered image can be huge -- trust our own source file and disable
# PIL's decompression-bomb safety limit.
Image.MAX_IMAGE_PIXELS = None

PDF_PATH = "data/sample_pitch_deck.pdf"

import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MIN_TEXT_LENGTH = 100


def extract_text_with_ocr(pdf_path: str, page_number: int) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]

    page_rect = page.rect
    strip_height = 2000  # points, processed in chunks to limit peak memory
    full_text = ""

    y = 0
    while y < page_rect.height:
        clip_rect = fitz.Rect(0, y, page_rect.width, min(y + strip_height, page_rect.height))
        pix = page.get_pixmap(dpi=100, clip=clip_rect, colorspace=fitz.csGRAY)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        strip_text = pytesseract.image_to_string(image)
        full_text += strip_text + "\n"

        image.close()
        pix = None
        y += strip_height

    doc.close()
    return full_text

def extract_text_from_pdf(path: str) -> str:
    full_text = ""
    with pdfplumber.open(path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages")
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            char_count = len(page_text.strip()) if page_text else 0

            if char_count >= MIN_TEXT_LENGTH:
                full_text += f"\n--- Page {i} (text layer, {char_count} chars) ---\n{page_text}"
            else:
                print(f"Page {i}: text layer too short ({char_count} chars), running OCR...")
                ocr_text = extract_text_with_ocr(path, i)
                full_text += f"\n--- Page {i} (OCR) ---\n{ocr_text}"

    return full_text


def extract_dollar_amounts(text: str) -> list:
    pattern = r"\$[\d,.]+\s?[BMK]?(?:/\w+)?"
    return re.findall(pattern, text)


def extract_percentages(text: str) -> list:
    pattern = r"\d+(?:\.\d+)?%"
    return re.findall(pattern, text)


def extract_user_counts(text: str) -> list:
    pattern = r"(?<!\$)\b\d+(?:\.\d+)?[BMK]\b"
    return re.findall(pattern, text)


def extract_team_mentions(text: str) -> list:
    # Matches the common pitch-deck format "Name | Title" e.g.
    # "Piotr Dabkowski | CTO" -- captures both name and role
    pattern = r"([A-Z][a-z]+\s[A-Z][a-z]+)\s*\|\s*(CEO|CTO|COO|CFO|Founder|Co-founder|President)"
    matches = re.findall(pattern, text)
    return [f"{name} ({title})" for name, title in matches]


if __name__ == "__main__":
    raw_text = extract_text_from_pdf(PDF_PATH)

    print("\n=== EXTRACTED DOLLAR AMOUNTS ===")
    print(extract_dollar_amounts(raw_text))

    print("\n=== EXTRACTED PERCENTAGES ===")
    print(extract_percentages(raw_text))

    print("\n=== EXTRACTED SCALE NUMBERS (users/market size, e.g. 96M, 4.6B) ===")
    print(extract_user_counts(raw_text))

    print("\n=== EXTRACTED TEAM/FOUNDER MENTIONS ===")
    print(extract_team_mentions(raw_text))

    with open("reports/pdf_extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    print("\nSaved full extracted text to reports/pdf_extracted_text.txt")