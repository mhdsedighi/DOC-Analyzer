import fitz  # PyMuPDF
import json
import base64
import os
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

        # Extract vector images with filtering and merging
        drawings = page.get_drawings()
        print(f"Found {len(drawings)} vector graphics on page {page_num + 1}.")

        filtered_drawings = []
        for drawing in drawings:
            rect = drawing["rect"]
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            if width * height > 10000:  # Ignore very small vector objects
                filtered_drawings.append(rect)

        if len(filtered_drawings) > 10:  # Limit number of extractions per page
            print(f"Too many vector objects on page {page_num + 1}, limiting to 10 largest.")
            filtered_drawings = sorted(filtered_drawings, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)[
                                :10]

        merged_vectors = []
        for rect in filtered_drawings:
            merged = False
            for i, m_rect in enumerate(merged_vectors):
                if (abs(m_rect[0] - rect[0]) < 10 and abs(m_rect[1] - rect[1]) < 10):
                    merged_vectors[i] = (
                    min(m_rect[0], rect[0]), min(m_rect[1], rect[1]), max(m_rect[2], rect[2]), max(m_rect[3], rect[3]))
                    merged = True
                    break
            if not merged:
                merged_vectors.append(rect)

        print(f"After filtering and merging, extracting {len(merged_vectors)} vector graphics.")

        for draw_index, rect in enumerate(merged_vectors):
            print(f"Extracting vector graphic {draw_index + 1} at {rect}...")
            pix = page.get_pixmap(clip=rect)
            image_bytes = pix.tobytes("png")
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            if len(encoded_image) >= 3000: # ignoring simple images (like page border lines)
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
