import os
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
import win32com.client
import fitz
import pdfplumber
from PIL import Image
from io import BytesIO
import base64
import pytesseract

# Function to extract text and images from a PDF file
def extract_content_from_pdf(pdf_path, do_read_image):
    text_content = []  # Stores extracted text per page
    image_content = []  # Stores extracted images per page if enabled
    unreadable_pages = 0
    total_pages = 0
    word_count = 0
    has_text = False
    used_ocr = "NO"

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            doc = fitz.open(pdf_path)

            for page_num in range(total_pages):
                page_text = ""
                image_index = 0

                # Extract text and image placeholders using PyMuPDF
                page = doc.load_page(page_num)
                blocks = page.get_text("dict")["blocks"]

                print(f"\nProcessing Page {page_num + 1}...")
                print(f"Number of blocks: {len(blocks)}")

                for block in blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text += span["text"]
                            page_text += "\n"
                    elif block["type"] == 1 and do_read_image:  # Image block
                        print(f"Found image block on page {page_num + 1}")
                        image_index += 1
                        page_text += f"[IMAGE_{image_index}]\n"

                        # Extract the image
                        xref = block["xref"]
                        print(f"Image XREF: {xref}")
                        base_image = doc.extract_image(xref)
                        print(f"Base image keys: {base_image.keys()}")  # Debug: Check keys in base_image

                        if "image" in base_image:
                            image_bytes = base_image["image"]
                            print(f"Image size: {len(image_bytes)} bytes")  # Debug: Check image size

                            # Convert image bytes to base64
                            image = Image.open(BytesIO(image_bytes))
                            if image.mode == "CMYK":
                                image = image.convert("RGB")
                            buffered = BytesIO()
                            image.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            image_content.append({"page": page_num + 1, "index": image_index, "content": img_base64})
                            print(f"Image {image_index} extracted and encoded as base64")
                        else:
                            print(f"Warning: 'image' key not found in base_image for XREF {xref}")

                # Append the page text to text_content
                if page_text.strip():
                    text_content.append({"page": page_num + 1, "content": page_text})
                    word_count += len(page_text.split())
                    has_text = True
                else:
                    unreadable_pages += 1

        # Handle scanned PDFs (no text found)
        if not has_text:
            print("\nNo text found. Applying OCR to all pages...")
            for page_num in range(total_pages):
                fitz_page = doc.load_page(page_num)
                pix = fitz_page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    text_content.append({"page": page_num + 1, "content": ocr_text})
                    word_count += len(ocr_text.split())
                    has_text = True
                    used_ocr = "English"

        # Calculate readable percentage
        readable_percentage = 100 - (unreadable_pages / total_pages * 100) if total_pages > 0 else 100
        return text_content, image_content, word_count, readable_percentage, used_ocr

    except Exception as e:
        print(f"Extraction failed: {e}")
        return [], [], 0, 0, "NO"