import os
import tkinter as tk
from tkinter import filedialog, scrolledtext
import PyPDF2
import ollama

# Set the model name and temperature
MODEL_NAME = "DeepSeek-R1:1.5b"  # Replace with your desired model
TEMPERATURE = 0.7  # Adjust the temperature as needed

# Initialize Ollama
ollama.pull(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def analyze_pdfs_in_folder(folder_path):
    """Analyze all PDFs in the folder and return their combined text."""
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            all_text += extract_text_from_pdf(pdf_path) + "\n\n"
    return all_text

def chat_with_ai(messages):
    """Send messages to the AI model and get a response."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"temperature": TEMPERATURE}
    )
    return response["message"]["content"]

def on_submit():
    """Handle the submit button click."""
    user_input = input_text.get("1.0", tk.END).strip()
    if user_input:
        # Add the user's message to the chat history
        messages.append({"role": "user", "content": user_input})
        
        # Get the AI's response
        response = chat_with_ai(messages)
        
        # Add the AI's response to the chat history
        messages.append({"role": "assistant", "content": response})
        
        # Display the conversation in the output text area
        output_text.insert(tk.END, f"You: {user_input}\nAI: {response}\n\n")
        input_text.delete("1.0", tk.END)

def select_folder():
    """Open a folder dialog and analyze PDFs in the selected folder."""
    global pdf_text, messages
    folder_path = filedialog.askdirectory()
    if folder_path:
        pdf_text = analyze_pdfs_in_folder(folder_path)
        output_text.insert(tk.END, f"Analyzed PDFs in: {folder_path}\n\n")
        
        # Initialize the chat with the PDF text as context
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. The following text is extracted from PDFs: {pdf_text}"}
        ]

# Initialize the main window
root = tk.Tk()
root.title("PDF Chat with AI")

# Create a frame for the input and output
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Output text area
output_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=20)
output_text.pack(pady=10)

# Input text area
input_text = tk.Text(frame, wrap=tk.WORD, width=80, height=5)
input_text.pack(pady=10)

# Submit button
submit_button = tk.Button(frame, text="Submit", command=on_submit)
submit_button.pack(pady=5)

# Folder selection button
folder_button = tk.Button(frame, text="Select Folder with PDFs", command=select_folder)
folder_button.pack(pady=5)

# Global variables
pdf_text = ""  # Stores the extracted text from PDFs
messages = []  # Stores the chat history

# Start the main loop
root.mainloop()
