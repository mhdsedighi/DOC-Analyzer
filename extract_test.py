import fitz  # PyMuPDF
import json
import base64
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from io import BytesIO
import pytesseract


def extract_pdf(pdf_path):
    # Determines whether the PDF has selectable text. Calls the appropriate function.
    doc = fitz.open(pdf_path)
    has_text = any(len(page.get_text("text").strip()) > 5 or len(page.get_text("words")) > 5 for page in doc)  # Improved detection
    if has_text:
        print("Detected formatted text in PDF. Using extract_formatted_pdf...")
        return extract_formatted_pdf(pdf_path)
    else:
        print("No selectable text found. Using OCR-based extraction...")
        return extract_printed_pdf(pdf_path)
        

def extract_formatted_pdf(pdf_path):
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

    return text_content, images_content


def extract_printed_pdf(pdf_path):

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text_content = ""
    images_content = []
    img_dir = "imgextract"
    os.makedirs(img_dir, exist_ok=True)

    print("Opening PDF file...")
    doc = fitz.open(pdf_path)
    print(f"PDF contains {len(doc)} pages.")
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_index = 0

    for page_num, page in enumerate(doc):
        print(f"Processing page {page_num + 1}...")
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply OCR to extract text
        page_text = pytesseract.image_to_string(gray)
        text_content += page_text + "\n"

    return text_content, images_content


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


def save_to_json(text, images, output_path):
    print("Saving extracted data to JSON file...")
    data = {
        "text": text,
        "images": images
    }

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    pdf_file = "test.pdf"
    json_file = "test.json"

    print("Starting extraction process...")
    text, images = extract_pdf(pdf_file)
    print(f"Extraction complete. Extracted {len(images)} images (including vector images).")
    save_to_json(text, images, json_file)
    print(f"Process finished successfully. JSON saved at {json_file}")
