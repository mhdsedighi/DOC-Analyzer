import fitz  # PyMuPDF
import json
import base64
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from io import BytesIO
import pytesseract
import langid
import pycountry
import io
from multiprocessing import Pool, cpu_count, Manager


def extract_content_from_pdf(pdf_path,do_read_image):
    # Determines whether the PDF has selectable text. Calls the appropriate function.
    doc = fitz.open(pdf_path)
    has_text = any(
        len(page.get_text("text").strip()) > 5 or len(page.get_text("words")) > 5 for page in doc)  # Improved detection
    if has_text:
        print("Detected formatted text in PDF. Using extract_formatted_pdf...")
        return extract_formatted_pdf(pdf_path,do_read_image)
    else:
        print("No selectable text found. Using OCR-based extraction...")
        return extract_printed_pdf(pdf_path)


def extract_formatted_pdf(pdf_path,do_read_image):
    text_content = ""
    images_content = []
    img_dir = "imgextract"
    os.makedirs(img_dir, exist_ok=True)

    print("Opening PDF file...")
    doc = fitz.open(pdf_path)
    print(f"PDF contains {len(doc)} pages.")
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    image_index = 0
    extracted_boxes = []  # Track bounding boxes of extracted images

    for page_num, page in enumerate(doc):
        print(f"Processing page {page_num + 1}...")
        page_text = page.get_text("text")
        page_elements = []  # Store text and image elements with their positions

        # Extract text blocks with their positions
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, block_no, block_type = block
            if text.strip():  # Only consider non-empty text blocks
                page_elements.append({
                    "type": "text",
                    "bbox": (x0, y0, x1, y1),
                    "content": text
                })
        if do_read_image:
            # Extract raster images
            images = page.get_images(full=True)
            print(f"Found {len(images)} raster images on page {page_num + 1}.")

            for img_index, img in enumerate(images):
                xref = img[0]
                print(f"Extracting raster image {img_index + 1} with xref {xref}...")
                base_image = doc.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

                    if len(encoded_image) >= 3000:
                        # Save the image and track its bounding box
                        images_content.append(encoded_image)
                        image_filename = os.path.join(img_dir, f"{pdf_name}_img_{image_index}.png")
                        with open(image_filename, "wb") as img_file:
                            img_file.write(image_bytes)
                        print(f"Raster image {img_index + 1} saved as {image_filename}.")

                        # Record the bounding box of the raster image
                        bbox = (0, 0, page.rect.width, page.rect.height)  # Full page for raster images
                        extracted_boxes.append(bbox)
                        page_elements.append({
                            "type": "image",
                            "bbox": bbox,
                            "content": f"[IMAGE_{image_index}]"
                        })
                        image_index += 1
                    else:
                        print(f"Raster image {img_index + 1} skipped due to small size.")

            # OpenCV-based vector detection with text exclusion
            print("Rendering page for vector detection...")
            pix = page.get_pixmap()
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            # Mask text areas in the image
            for x0, y0, x1, y1 in [elem["bbox"] for elem in page_elements if elem["type"] == "text"]:
                cv2.rectangle(gray, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), -1)

            # Apply adaptive thresholding for better edge detection
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Apply morphological closing to merge close elements
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Detected {len(contours)} vector elements using OpenCV.")

            bounding_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # Ignore very thin text-like shapes
                if w / h > 10 or h / w > 10:
                    continue

                bounding_boxes.append([x, y, x + w, y + h])

            # Apply DBSCAN clustering to merge nearby bounding boxes
            if len(bounding_boxes) > 0:
                clustering = DBSCAN(eps=20, min_samples=1).fit(bounding_boxes)
                labels = clustering.labels_
                merged_boxes = {}

                for i, label in enumerate(labels):
                    if label not in merged_boxes:
                        merged_boxes[label] = bounding_boxes[i]
                    else:
                        merged_boxes[label][0] = min(merged_boxes[label][0], bounding_boxes[i][0])
                        merged_boxes[label][1] = min(merged_boxes[label][1], bounding_boxes[i][1])
                        merged_boxes[label][2] = max(merged_boxes[label][2], bounding_boxes[i][2])
                        merged_boxes[label][3] = max(merged_boxes[label][3], bounding_boxes[i][3])

                final_boxes = list(merged_boxes.values())
            else:
                final_boxes = []

            print(f"After merging, extracting {len(final_boxes)} vector graphics.")

            for draw_index, rect in enumerate(final_boxes):
                # Check if this vector graphic overlaps with an already extracted image
                is_duplicate = False
                for extracted_box in extracted_boxes:
                    if is_overlap(rect, extracted_box):
                        is_duplicate = True
                        break

                if is_duplicate:
                    print(f"Vector graphic {draw_index + 1} skipped as it overlaps with an already extracted image.")
                    continue

                print(f"Extracting vector graphic {draw_index + 1} at {rect}...")
                pix = page.get_pixmap(clip=rect)
                image_bytes = pix.tobytes("png")
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")

                if len(encoded_image) >= 3000:
                    images_content.append(encoded_image)
                    image_filename = os.path.join(img_dir, f"{pdf_name}_vector_{image_index}.png")
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                    print(f"Vector image {draw_index + 1} saved as {image_filename}.")

                    # Record the bounding box of the vector graphic
                    extracted_boxes.append(rect)
                    page_elements.append({
                        "type": "image",
                        "bbox": rect,
                        "content": f"[IMAGE_{image_index}]"
                    })
                    image_index += 1
                else:
                    print(f"Vector image {draw_index + 1} skipped due to small size.")

        # Sort elements by their vertical (y) and horizontal (x) positions
        page_elements.sort(key=lambda elem: (elem["bbox"][1], elem["bbox"][0]))  # Sort by y, then x

        # Build the text with inline placeholders
        page_text_with_placeholders = ""
        for elem in page_elements:
            if elem["type"] == "text":
                page_text_with_placeholders += elem["content"]
            elif elem["type"] == "image":
                page_text_with_placeholders += elem["content"]

        text_content += page_text_with_placeholders + "\n"

    word_count = len(text_content.split())
    return text_content, images_content,word_count,100,"NO"


def extract_printed_pdf(pdf_path, tesseract_path=None):
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    # Get list of installed Tesseract languages
    installed_langs = pytesseract.get_languages(config='')
    print(f"Installed Tesseract languages: {installed_langs}")

    try:
        print("Opening PDF file...")
        doc = fitz.open(pdf_path)
        print(f"PDF contains {len(doc)} pages.")

        # Convert the PDF to bytes (shared across processes)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Extract a text sample for language detection
        sample_text = ""
        sample_pages = min(3, len(doc))

        for page_num in range(sample_pages):
            page = doc.load_page(page_num)
            img = np.frombuffer(page.get_pixmap().samples, dtype=np.uint8).reshape(page.get_pixmap().h,
                                                                                   page.get_pixmap().w,
                                                                                   page.get_pixmap().n)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            page_text, _ = perform_ocr(gray, "eng")
            sample_text += page_text + " "

        detected_lang = detect_language(sample_text, installed_langs)
        print(f"Detected language: {detected_lang}")

        # Use multiprocessing with an initializer to pass pdf_bytes
        with Manager() as manager:
            shared_pdf_bytes = manager.Value(bytes, pdf_bytes)
            with Pool(processes=cpu_count(), initializer=init_worker, initargs=(shared_pdf_bytes,)) as pool:
                results = pool.starmap(process_ocr_page, [(page_num, detected_lang) for page_num in range(len(doc))])

        text_content = [result[0] for result in results]
        # images_content = [result[1] for result in results]

        word_count = len(text_content.split())
        return text_content, [],word_count,100,detected_lang

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [],0,0,""


# Global variable for worker processes
worker_pdf = None


def init_worker(shared_pdf_bytes):
    """Initializer function to load the PDF once per worker."""
    global worker_pdf
    worker_pdf = fitz.open(stream=io.BytesIO(shared_pdf_bytes.get()))


def process_ocr_page(page_num, lang):
    """Process a single page using the globally stored PDF object."""
    global worker_pdf
    page = worker_pdf.load_page(page_num)
    print(f"Processing page {page_num + 1}...")
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    page_text, confidence = perform_ocr(gray, lang)
    return page_text, []


def perform_ocr(image, lang):
    """Perform OCR on the given image using Tesseract with advanced preprocessing."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Apply adaptive thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # Define custom configuration
    custom_config = f'--oem 3 --psm 6 -l {lang}'

    # Perform OCR
    data = pytesseract.image_to_data(denoised, config=custom_config, output_type=pytesseract.Output.DICT)
    words = [word for word in data["text"] if word.strip()]
    valid_conf = [c for c in data["conf"] if c != -1]
    text = " ".join(words)
    confidence = sum(valid_conf) / len(valid_conf) if valid_conf else 0

    return text, confidence


def detect_language(text, installed_langs):
    """Detect the language of the text, ensuring it is a valid Tesseract language."""
    lang_code, _ = langid.classify(text)

    # Map ISO 639-1 code to ISO 639-3 using pycountry
    try:
        lang = pycountry.languages.get(alpha_2=lang_code)
        tesseract_lang = lang.alpha_3
    except AttributeError:
        tesseract_lang = "eng"  # Fallback to English if mapping fails

    # Ensure the detected language is installed in Tesseract
    if tesseract_lang in installed_langs:
        return tesseract_lang
    else:
        print(f"Detected language '{tesseract_lang}' is not installed in Tesseract. Falling back to English.")
        return "eng"


def is_overlap(box1, box2, threshold=0.8):
    """
    Check if two bounding boxes overlap significantly.
    box1 and box2 are tuples/lists of (x0, y0, x1, y1).
    threshold: Minimum overlap ratio to consider as a duplicate.
    """
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    # Calculate intersection area
    x0 = max(x0_1, x0_2)
    y0 = max(y0_1, y0_2)
    x1 = min(x1_1, x1_2)
    y1 = min(y1_1, y1_2)

    if x1 <= x0 or y1 <= y0:
        return False  # No overlap

    intersection_area = (x1 - x0) * (y1 - y0)
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

    # Calculate overlap ratio
    overlap_ratio = intersection_area / min(area1, area2)
    return overlap_ratio >= threshold
