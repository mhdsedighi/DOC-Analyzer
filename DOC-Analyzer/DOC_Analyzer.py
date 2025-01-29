import os
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import PyPDF2
import ollama
from pytesseract import image_to_string
from pdf2image import convert_from_path
from PIL import Image

# Set the model name and temperature
MODEL_NAME = "DeepSeek-R1:1.5b"  # Replace with your desired model
TEMPERATURE = 0.7  # Adjust the temperature as needed

# File paths for saving last folder and cache
LAST_FOLDER_FILE = "last_folder.txt"
PDF_CACHE_FILE = "pdf_cache.json"

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

# Load the PDF cache
def load_pdf_cache():
    if os.path.exists(PDF_CACHE_FILE):
        with open(PDF_CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

# Save the PDF cache
def save_pdf_cache(cache):
    with open(PDF_CACHE_FILE, "w") as file:
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

    # Calculate the percentage of unreadable pages
    unreadable_percentage = (unreadable_pages / total_pages * 100) if total_pages > 0 else 0

    # Count the number of words
    word_count = len(text.split())

    return text, word_count, unreadable_percentage

# Function to analyze all PDFs in a folder
def analyze_pdfs(folder_path):
    cache = load_pdf_cache()
    all_text = ""
    new_files_analyzed = 0  # Track the number of new files analyzed

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            last_modified = os.path.getmtime(pdf_path)
            
            # Check if the file is in the cache and hasn't been modified
            if pdf_path in cache and cache[pdf_path]["last_modified"] == last_modified:
                text = cache[pdf_path]["text"]
                word_count = cache[pdf_path]["word_count"]
                unreadable_percentage = cache[pdf_path]["unreadable_percentage"]
            else:
                # Extract text and update the cache
                text, word_count, unreadable_percentage = extract_text_from_pdf(pdf_path)
                cache[pdf_path] = {
                    "last_modified": last_modified,
                    "text": text,
                    "word_count": word_count,
                    "unreadable_percentage": unreadable_percentage
                }
                new_files_analyzed += 1  # Increment the counter for new files
            
            all_text += f"--- {filename} ---\n{text}\n\n"
            chat_history.config(state=tk.NORMAL)
            chat_history.insert(tk.END, f"Analyzed: {filename}\n")
            chat_history.insert(tk.END, f"Word count: {word_count}\n")
            chat_history.insert(tk.END, f"Unreadable pages: {unreadable_percentage:.2f}%\n\n")
            chat_history.config(state=tk.DISABLED)
    
    # Save the updated cache
    save_pdf_cache(cache)
    return all_text, new_files_analyzed

# Function to handle the chat with the AI
def chat_with_ai():
    user_input = user_input_box.get("1.0", tk.END).strip()
    if user_input:
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, f"You: {user_input}\n")
        
        # Combine the PDF text with the user input
        full_prompt = f"{pdf_text}\n\nUser: {user_input}"
        
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

# Function to set the folder path
def set_folder_path():
    folder_path = folder_path_entry.get().strip()
    if not folder_path:
        messagebox.showwarning("Warning", "Please enter a folder path.")
        return
    
    if not os.path.isdir(folder_path):
        messagebox.showerror("Error", "Invalid folder path.")
        return
    
    global pdf_text
    pdf_text, new_files_analyzed = analyze_pdfs(folder_path)
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"PDFs analyzed. {new_files_analyzed} new files were processed.\n")
    chat_history.insert(tk.END, "You can now chat with the AI.\n")
    chat_history.config(state=tk.DISABLED)
    
    # Save the folder path
    save_last_folder(folder_path)

# Create the main window
root = tk.Tk()
root.title("PDF Analyzer and AI Chat")

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
analyze_button = tk.Button(root, text="Analyze PDFs", command=set_folder_path)
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

# Run the application
root.mainloop()