import os
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import PyPDF2
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
import ollama
from pytesseract import image_to_string
from pdf2image import convert_from_path
from PIL import Image

# Set the model name and temperature
MODEL_NAME = "DeepSeek-R1:1.5b"  # Replace with your desired model
TEMPERATURE = 0.7  # Adjust the temperature as needed

# File paths for saving last folder and cache
LAST_FOLDER_FILE = "last_folder.txt"
DOCUMENT_CACHE_FILE = "document_cache.json"

# Supported file extensions
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".xlsx", ".pptx"]

# Load the last used folder path
def load_last_folder():
    if os.path.exists(LAST_FOLDER_FILE):
        with open(LAST_FOLDER_FILE, "r") as file:
            return file.read().strip()
    return ""

# Save the last used folder path
def save_last_folder(folder_path):
    with open(LAST_FOLDER_FILE, "w") as file:
        file.write(folder_path)

# Load the document cache
def load_document_cache():
    if os.path.exists(DOCUMENT_CACHE_FILE):
        with open(DOCUMENT_CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

# Save the document cache
def save_document_cache(cache):
    with open(DOCUMENT_CACHE_FILE, "w") as file:
        json.dump(cache, file, indent=4)

# Function to extract text from a PDF file using PyPDF2 and OCR
def extract_text_from_pdf(pdf_path):
    text = ""
    unreadable_pages = 0
    total_pages = 0

    try:
        # Try extracting text using PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    unreadable_pages += 1
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")

    # If PyPDF2 fails or some pages are unreadable, use OCR
    if unreadable_pages > 0:
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            for image in images:
                ocr_text = image_to_string(image)
                text += ocr_text + "\n"
        except Exception as e:
            print(f"OCR extraction failed: {e}")

    # Calculate the percentage of readable content
    readable_percentage = 100 - (unreadable_pages / total_pages * 100) if total_pages > 0 else 100

    # Count the number of words
    word_count = len(text.split())

    return text, word_count, readable_percentage

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        word_count = len(text.split())
        return text, word_count, 100  # Assume 100% readable for DOCX
    except Exception as e:
        print(f"DOCX extraction failed: {e}")
        return "", 0, 0  # Assume 0% readable if extraction fails

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
        word_count = len(text.split())
        return text, word_count, 100  # Assume 100% readable for TXT
    except Exception as e:
        print(f"TXT extraction failed: {e}")
        return "", 0, 0  # Assume 0% readable if extraction fails

# Function to extract text from an XLSX file
def extract_text_from_xlsx(xlsx_path):
    try:
        workbook = load_workbook(xlsx_path)
        text = ""
        for sheet in workbook:
            for row in sheet.iter_rows(values_only=True):
                text += " ".join([str(cell) for cell in row if cell is not None]) + "\n"
        word_count = len(text.split())
        return text, word_count, 100  # Assume 100% readable for XLSX
    except Exception as e:
        print(f"XLSX extraction failed: {e}")
        return "", 0, 0  # Assume 0% readable if extraction fails

# Function to extract text from a PPTX file
def extract_text_from_pptx(pptx_path):
    try:
        presentation = Presentation(pptx_path)
        text = ""
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        word_count = len(text.split())
        return text, word_count, 100  # Assume 100% readable for PPTX
    except Exception as e:
        print(f"PPTX extraction failed: {e}")
        return "", 0, 0  # Assume 0% readable if extraction fails

# Function to extract text from a document based on its file type
def extract_text_from_document(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    elif file_path.endswith(".xlsx"):
        return extract_text_from_xlsx(file_path)
    elif file_path.endswith(".pptx"):
        return extract_text_from_pptx(file_path)
    else:
        return "", 0, 0  # Unsupported file type

# Function to analyze all documents in a folder
def analyze_documents(folder_path):
    cache = load_document_cache()
    all_text = ""
    new_files_analyzed = 0  # Track the number of new files analyzed

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in SUPPORTED_EXTENSIONS:
            last_modified = os.path.getmtime(file_path)
            
            # Check if the file is in the cache and hasn't been modified
            if file_path in cache and cache[file_path]["last_modified"] == last_modified:
                text = cache[file_path]["text"]
                word_count = cache[file_path]["word_count"]
                readable_percentage = cache[file_path]["readable_percentage"]
            else:
                # Extract text and update the cache
                text, word_count, readable_percentage = extract_text_from_document(file_path)
                cache[file_path] = {
                    "last_modified": last_modified,
                    "text": text,
                    "word_count": word_count,
                    "readable_percentage": readable_percentage
                }
                new_files_analyzed += 1  # Increment the counter for new files
            
            all_text += f"--- {filename} ---\n{text}\n\n"
            chat_history.config(state=tk.NORMAL)
            chat_history.insert(tk.END, f"Analyzed: {filename}\n")
            chat_history.insert(tk.END, f"Word count: {word_count}\n")
            chat_history.insert(tk.END, f"Readable content: {readable_percentage:.2f}%\n\n")
            chat_history.config(state=tk.DISABLED)
    
    # Save the updated cache
    save_document_cache(cache)
    return all_text, new_files_analyzed

# Function to handle the chat with the AI
def chat_with_ai():
    user_input = user_input_box.get("1.0", tk.END).strip()
    if user_input:
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, f"You: {user_input}\n")
        
        # Combine the document text with the user input
        full_prompt = f"{document_text}\n\nUser: {user_input}"
        
        try:
            # Send the prompt to the AI
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                options={"temperature": TEMPERATURE}
            )
            
            ai_response = response['message']['content']
            chat_history.insert(tk.END, f"AI: {ai_response}\n")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        
        chat_history.config(state=tk.DISABLED)
        user_input_box.delete("1.0", tk.END)

# Function to clear the chat history box
def clear_chat_history():
    chat_history.config(state=tk.NORMAL)
    chat_history.delete("1.0", tk.END)
    chat_history.config(state=tk.DISABLED)

# Function to set the folder path
def set_folder_path():
    folder_path = folder_path_entry.get().strip()
    if not folder_path:
        messagebox.showwarning("Warning", "Please enter a folder path.")
        return
    
    if not os.path.isdir(folder_path):
        messagebox.showerror("Error", "Invalid folder path.")
        return
    
    global document_text
    document_text, new_files_analyzed = analyze_documents(folder_path)
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"Documents analyzed. {new_files_analyzed} new files were processed.\n")
    chat_history.insert(tk.END, "You can now chat with the AI.\n")
    chat_history.config(state=tk.DISABLED)
    
    # Save the folder path
    save_last_folder(folder_path)

# Create the main window
root = tk.Tk()
root.title("Document Analyzer and AI Chat")

# Folder path entry
folder_path_entry = tk.Entry(root, width=50)
folder_path_entry.grid(row=0, column=0, padx=10, pady=10)

# Load the last used folder path
last_folder = load_last_folder()
if last_folder:
    folder_path_entry.insert(0, last_folder)

# Browse button
browse_button = tk.Button(root, text="Browse", command=lambda: folder_path_entry.insert(0, filedialog.askdirectory()))
browse_button.grid(row=0, column=1, padx=10, pady=10)

# Analyze button
analyze_button = tk.Button(root, text="Analyze Documents", command=set_folder_path)
analyze_button.grid(row=0, column=2, padx=10, pady=10)

# Chat history display
chat_history = scrolledtext.ScrolledText(root, width=80, height=20, state=tk.DISABLED)
chat_history.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

# User input box
user_input_box = tk.Text(root, width=60, height=3)
user_input_box.grid(row=2, column=0, padx=10, pady=10)

# Send button
send_button = tk.Button(root, text="Send", command=chat_with_ai)
send_button.grid(row=2, column=1, padx=10, pady=10)

# Clear button
clear_button = tk.Button(root, text="Clear", command=clear_chat_history)
clear_button.grid(row=2, column=2, padx=10, pady=10)

# Run the application
root.mainloop()