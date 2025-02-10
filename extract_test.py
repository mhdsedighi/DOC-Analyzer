import fitz  # PyMuPDF
import json
import base64
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from io import BytesIO


def extract_text_and_images(pdf_path):
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
        page_text = page.get_text("text")
        page_text_with_placeholders = ""
        last_pos = 0

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
                    images_content.append(encoded_image)
                    image_filename = os.path.join(img_dir, f"{pdf_name}_img_{image_index}.png")
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                    print(f"Raster image {img_index + 1} saved as {image_filename}.")
                    page_text_with_placeholders += page_text[last_pos:] + f" [IMAGE_{image_index}] "
                    last_pos = len(page_text)
                    image_index += 1
                else:
                    print(f"Raster image {img_index + 1} skipped due to small size.")

        # Extract text bounding boxes and create a mask
        text_bboxes = [bbox[:4] for bbox in page.get_text("blocks")]

        print("Rendering page for vector detection...")
        pix = page.get_pixmap()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Create a mask for text regions
        text_mask = np.zeros_like(edges)
        for bbox in text_bboxes:
            x0, y0, x1, y1 = map(int, bbox)
            cv2.rectangle(text_mask, (x0, y0), (x1, y1), 255, thickness=cv2.FILLED)

        # Apply the mask to remove text regions from edges
        edges[text_mask == 255] = 0

        # Apply morphological closing to merge close elements
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Detected {len(contours)} vector elements using OpenCV.")

        bounding_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 10000:  # Ignore very small vector objects
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
                page_text_with_placeholders += page_text[last_pos:] + f" [IMAGE_{image_index}] "
                last_pos = len(page_text)
                image_index += 1
            else:
                print(f"Vector image {draw_index + 1} skipped due to small size.")

        page_text_with_placeholders += page_text[last_pos:] + "\n"
        text_content += page_text_with_placeholders

    return text_content, images_content


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
    text, images = extract_text_and_images(pdf_file)
    print(f"Extraction complete. Extracted {len(images)} images (including vector images).")
    save_to_json(text, images, json_file)
    print(f"Process finished successfully. JSON saved at {json_file}")
