import os
import json

# Define the cache folder and ensure it exists
if not os.path.exists("cache"):
    os.makedirs("cache")

# File paths for saving user data and cache
USER_DATA_FILE = os.path.join("cache", "user_data.json")
DOCUMENT_CACHE_FILE = os.path.join("cache", "document_cache.json")

# Load user data (last folder path, last selected model, and temperature)
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r", encoding="utf-8") as file:
            user_data = json.load(file)
            return {
                "last_folder": user_data.get("last_folder", ""),
                "last_model": user_data.get("last_model", ""),
                "temperature": user_data.get("temperature", 0.7),
                "do_mention_page": user_data.get("do_mention_page", False),
                "do_read_image": user_data.get("do_read_image", False),
                "tesseract_folder": user_data.get("tesseract_folder", r"C:\Program Files\Tesseract-OCR"),  # Load tesseract_folder
            }
    return {
        "last_folder": "",
        "last_model": "",
        "temperature": 0.7,
        "do_mention_page": False,
        "do_read_image": False,
        "tesseract_folder": r"C:\Program Files\Tesseract-OCR",  # Default value
    }

# Save user data (last folder path, last selected model, temperature, and last 10 folders)
def save_user_data(last_folder=None, last_model=None, temperature=None, last_folders=None, do_mention_page=None, do_read_image=None, tesseract_folder=None):
    user_data = load_user_data()
    if last_folder is not None:
        user_data["last_folder"] = last_folder
    if last_model is not None:
        user_data["last_model"] = last_model
    if temperature is not None:
        user_data["temperature"] = temperature
    if last_folders is not None:
        user_data["last_folders"] = last_folders
    if do_mention_page is not None:
        user_data["do_mention_page"] = do_mention_page
    if do_read_image is not None:
        user_data["do_read_image"] = do_read_image
    if tesseract_folder is not None:
        user_data["tesseract_folder"] = tesseract_folder

    # Update the last 10 folders list if a new folder is added
    if last_folder and last_folder not in user_data.get("last_folders", []):
        if "last_folders" not in user_data:
            user_data["last_folders"] = []
        user_data["last_folders"].insert(0, last_folder)
        user_data["last_folders"] = user_data["last_folders"][:10]  # Keep only the last 10

    with open(USER_DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(user_data, file, indent=4, ensure_ascii=False)