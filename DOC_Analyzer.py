import os
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from tkinter import Menu
import enchant  # For spell checking
import ollama
from modules.file_read import extract_content_from_file

# Define the cache folder and ensure it exists
if not os.path.exists("cache"):
    os.makedirs("cache")

# File paths for saving user data and cache
USER_DATA_FILE = os.path.join("cache", "user_data.json")
DOCUMENT_CACHE_FILE = os.path.join("cache", "document_cache.json")

# Supported file extensions
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".pptx", ".ppt"]

chat_history_list = []  # List to store the conversation history

# Load user data (last folder path, last selected model, and temperature)
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            data = json.load(file)
            # Ensure the "last_folders" and "do_mention_page" keys exist in the loaded data
            if "last_folders" not in data:
                data["last_folders"] = []
            if "do_mention_page" not in data:
                data["do_mention_page"] = False  # Default value if not present
            if "do_read_image" not in data:
                data["do_read_image"] = False  # Default value if not present
            return data
    return {
        "last_folder": "",
        "last_folders": [],
        "last_model": "",
        "temperature": 0.7,
        "do_mention_page": False,  # Default value for the checkbox
        "do_read_image": False  # Default value for the checkbox
    }

# Save user data (last folder path, last selected model, temperature, and last 10 folders)
def save_user_data(last_folder=None, last_model=None, temperature=None, last_folders=None, do_mention_page=None ,do_read_image=None):
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

    # Update the last 10 folders list if a new folder is added
    if last_folder and last_folder not in user_data.get("last_folders", []):
        if "last_folders" not in user_data:
            user_data["last_folders"] = []
        user_data["last_folders"].insert(0, last_folder)
        user_data["last_folders"] = user_data["last_folders"][:10]  # Keep only the last 10

    with open(USER_DATA_FILE, "w") as file:
        json.dump(user_data, file, indent=4)

    # Update the last 10 folders list
    if last_folder:
        if last_folder in user_data["last_folders"]:
            user_data["last_folders"].remove(last_folder)  # Remove if already exists
        user_data["last_folders"].insert(0, last_folder)  # Add to the beginning
        user_data["last_folders"] = user_data["last_folders"][:10]  # Keep only the last 10

    with open(USER_DATA_FILE, "w") as file:
        json.dump(user_data, file, indent=4)

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

# Function to fetch installed Ollama models
def fetch_installed_models():
    try:
        models_response = ollama.list()  # Fetch the list of installed models
        # print("DEBUG: Response from ollama.list():", models_response)  # Debugging output
        
        models_list = models_response.models  # Access the 'models' attribute directly

        if not isinstance(models_list, list):
            raise ValueError("Unexpected response format: 'models' key is not a list")

        # Extract model names
        model_names = [getattr(model, "model", "Unknown") for model in models_list]

        return model_names

    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch models: {e}")
        return []

# Function to handle folder browsing
def browse_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_path_entry.delete(0, tk.END)  # Clear the entry widget
        folder_path_entry.insert(0, folder_path)  # Insert the new folder path
        update_folder_dropdown()  # Update the dropdown with the new path
        save_user_data(last_folder=folder_path)  #save the new folder path


# Function to read all documents in a folder
def read_documents(folder_path):
    document_text = [] #this appers to fix adding extra texts when changing folder
    cache = load_document_cache()
    new_files_read = 0  # Track the number of new files read
    document_images = []  # List to store images from documents

    # Set the introductory line based on the checkbox state. 4 Scenarios
    if not do_send_images:
        all_text = """You are analyzing extracted text from multiple documents (PDFs and DOCX files). Each text entry includes a file name and page number at the top of the text, structured as:
                        { "file": "filename.pdf", "page": X, "content": "text data" }.
                        Provide insights, summaries, or answers based on this textual content."""
    else:
        all_text = """You are analyzing extracted text, images (including both raster and vector types), and diagrams from multiple documents (PDFs and DOCX files). Each entry includes a file name and page number at the top, structured as:
                       { "file": "filename.pdf", "page": X, "content": "text or base64-encoded image or raw vector data (SVG)" }.
                        Some entries may contain base64-encoded raster images (e.g., PNG, JPEG), while others may contain raw vector data (e.g., SVG) for scalable images. Consider both text and these visual elements when generating insights, summaries, or answers."""
    if do_mention_var.get():
        all_text += """When providing insights, summaries, or answers, reference the file name and page number where the information was found."""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in SUPPORTED_EXTENSIONS:
            last_modified = os.path.getmtime(file_path)

            # Check if file is in cache and hasn't been modified
            if file_path in cache and cache[file_path]["last_modified"] == last_modified:
                cached_data = cache[file_path]
                text_content = cached_data["text_content"]
                word_count = cached_data["word_count"]
                readable_percentage = cached_data["readable_percentage"]
                image_content = cached_data.get("image_content", [])

                # **Re-extract if images were not previously stored and do_read_image is True**
                if do_read_image_var.get() and not image_content: # and not file_ext=="txt"
                    text_content, image_content, word_count, readable_percentage = extract_content_from_file(file_path,do_read_image_var.get())

                    # Update the cache with new images
                    cache[file_path]["image_content"] = image_content
                    save_document_cache(cache)
                    new_files_read += 1  # Increment new file count if reprocessed

            else:
                # Extract text and images, then update the cache
                text_content, image_content, word_count, readable_percentage = extract_content_from_file(file_path,do_send_images)

                cache[file_path] = {
                    "last_modified": last_modified,
                    "text_content": text_content,
                    "word_count": word_count,
                    "readable_percentage": readable_percentage,
                    "image_content": image_content,  # Cache images with their page numbers
                }
                save_document_cache(cache)
                new_files_read += 1  # Increment counter for new files

            # Add filename and extracted content to combined text
            all_text += f"--- Below is the content of a document with the name {filename} ---\n"
            for text_data in text_content:
                all_text += f"[Page {text_data['page']}]: {text_data['content']}\n\n"

            chat_history.config(state=tk.NORMAL)
            chat_history.insert(tk.END, f"Looked at: {filename}\n", "fileread_tag")
            chat_history.insert(tk.END, f"Word count: {word_count}\n", "fileread_tag")
            chat_history.insert(tk.END, f"Extracted images: {len(image_content)}\n\n", "fileread_tag")
            chat_history.insert(tk.END, f"Readable content: {readable_percentage:.2f}%\n\n", "fileread_tag")
            chat_history.config(state=tk.DISABLED)

            # Add images to document_images list if model supports images
            if do_send_images:
                document_images.extend(image_content)

    return all_text, new_files_read, document_images


# Function to handle the chat with the AI
def chat_with_ai():
    user_input = user_input_box.get("1.0", tk.END).strip()
    global previous_message
    global do_revise
    previous_message=user_input

    if do_revise:  #removing what has been revised from prompt
        if chat_history_list:
            chat_history_list.pop()
            chat_history_list.pop()
        do_revise=False

    if user_input:
        chat_history.config(state=tk.NORMAL)
        
        # Insert the user's question
        chat_history.insert(tk.END, "You: ", "user_tag")  # Tag for user text
        chat_history.insert(tk.END, f"{user_input}\n", "user_question")  # Tag for user question
        
        # Add the user's message to the chat history list
        chat_history_list.append({"role": "user", "content": user_input})
        
        # Combine the document text with the chat history
        full_prompt = f"{document_text}\n\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_list])
        
        try:
            # Send the prompt to the AI using the selected model
            selected_model = model_var.get()
            messages = [{"role": "user", "content": full_prompt}]

            # If the model supports images, include them in the messages
            if do_send_images:
                for img_base64 in document_images:
                    messages.append({"role": "user", "content": f"data:image/png;base64,{img_base64}"})

            response = ollama.chat(
                model=selected_model,
                messages=messages,
                options={"temperature": temperature_scale.get()}  # Use the slider value
            )
            
            ai_response = response['message']['content']
            chat_history.insert(tk.END, "AI: ", "ai_tag")  # Tag for "AI:" 
            chat_history.insert(tk.END, f"{ai_response}\n", "ai_response")  # Tag for AI response
            
            # Add a horizontal line after the AI's response
            chat_history.insert(tk.END, "---\n", "separator")  # Tag for the separator line
            
            # Add the AI's response to the chat history list
            chat_history_list.append({"role": "assistant", "content": ai_response})
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        
        chat_history.config(state=tk.DISABLED)
        user_input_box.delete("1.0", tk.END)

        chat_history.see(tk.END) #scrolling down
        
        # Save the last used model, folder path, temperature, and checkbox state
        save_user_data(
            folder_path_entry.get().strip(),
            selected_model,
            temperature_scale.get(),
            do_mention_page=do_mention_var.get(),
            do_read_image=do_read_image_var.get()
        )

# Function to clear the chat history box
def clear_chat_history():
    chat_history.config(state=tk.NORMAL)
    chat_history.delete("1.0", tk.END)
    chat_history.config(state=tk.DISABLED)
    chat_history_list.clear()  # Clear the chat history list

# Function to set the folder path
def set_folder_path():
    global document_text
    document_text = ""  # Clear the previous document text
    global document_images
    document_images= []

    folder_path = folder_path_entry.get().strip()
    if not folder_path:
        messagebox.showwarning("Warning", "Please enter a folder path.")
        return
    
    if not os.path.isdir(folder_path):
        messagebox.showerror("Error", "Invalid folder path.")
        return
    
    # Read documents from the new folder
    document_text, new_files_read, document_images =  read_documents(folder_path)
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"Documents reading finished. {new_files_read} new files were processed.\n")
    chat_history.insert(tk.END, "You can now chat with the A.I.\n")
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END) #scrolling down
    
    # Save the folder path, last used model, and temperature
    save_user_data(folder_path, model_var.get(), temperature_scale.get())

    # Update the dropdown with the latest folder paths
    update_folder_dropdown()

# Function to update the folder dropdown with the last 10 used paths
def update_folder_dropdown():
    user_data = load_user_data()
    folder_path_dropdown["values"] = user_data.get("last_folders", [])
    folder_path_dropdown.set(folder_path_entry.get().strip())  # Sync dropdown with entry

# Function to handle folder path selection from the dropdown
def on_folder_select(event):
    selected_path = folder_path_dropdown.get()
    folder_path_entry.delete(0, tk.END)
    folder_path_entry.insert(0, selected_path)

def delete_folder_path(event):
    selected_path = folder_path_dropdown.get()  # Get the currently selected path
    if selected_path:
        user_data = load_user_data()  # Load the current user data
        if selected_path in user_data["last_folders"]:
            user_data["last_folders"].remove(selected_path)  # Remove the selected path
            save_user_data(last_folders=user_data["last_folders"]) # Save the updated user data

            # If the last_folders list is now empty, remove the last_folder as well
            if not user_data["last_folders"]:
                user_data.pop("last_folder", None)  # Remove last_folder from memory
                folder_path_entry.delete(0, tk.END)  # Clear the folder path entry

            # Save the updated user data
            with open(USER_DATA_FILE, "w") as file:
                json.dump(user_data, file, indent=4)
            # Update the dropdown with the new list of folders
            folder_path_dropdown["values"] = user_data["last_folders"]
            # Clear the current selection in the dropdown
            folder_path_dropdown.set("")

            # Create a temporary popup
            popup = tk.Toplevel(root)
            popup.overrideredirect(True)  # Remove window decorations (no close button)
            popup.geometry("400x50+{}+{}".format(
                root.winfo_x() + root.winfo_width() // 2 - 50,  # Center horizontally
                root.winfo_y() + root.winfo_height() // 2 - 25  # Center vertically
            ))
            popup.configure(bg="black")  # Set background color

            # Add a label with the text "Deleted."
            label = tk.Label(popup, text="Folder Path Deleted!", fg="white", bg="black", font=("Arial", 12 ,"bold"))
            label.pack(pady=10)

            # Function to fade out the popup
            def fade_out(popup, alpha=1.0):
                if alpha <= 0:
                    popup.destroy()  # Close the popup
                    return
                popup.attributes("-alpha", alpha)  # Set transparency
                root.after(150, fade_out, popup, alpha - 0.1)  # Reduce alpha every 150ms

            # Start fading out after 500ms
            root.after(500, fade_out, popup)
  
# Function to check spelling and underline misspelled words
def check_spelling():
    user_input_box.tag_remove("misspelled", "1.0", "end")  # Clear previous misspelled tags
    text = user_input_box.get("1.0", "end-1c")  # Get the text from the input widget
    words = text.split()  # Split text into words
    start_index = "1.0"  # Start checking from the beginning

    for word in words:
        # Calculate the end index of the current word
        end_index = f"{start_index}+{len(word)}c"
        if is_english(word):
            if not spell_checker.check(word):
                user_input_box.tag_add("misspelled", start_index, end_index)  # Tag misspelled word
        else:
            pass
        
        # Move the start index to the next word
        start_index = f"{end_index}+1c"

# Function to show spelling suggestions on right-click
def show_suggestions(event):
    # Get the word under the cursor
    word_start = user_input_box.index(f"@{event.x},{event.y} wordstart")
    word_end = user_input_box.index(f"@{event.x},{event.y} wordend")
    word = user_input_box.get(word_start, word_end)

    # Check if the word is misspelled
    if not spell_checker.check(word):
        # Get suggestions for the misspelled word
        suggestions = spell_checker.suggest(word)
        
        # Create a right-click menu
        menu = Menu(root, tearoff=0)
        for suggestion in suggestions:
            menu.add_command(label=suggestion, command=lambda s=suggestion: replace_word(word_start, word_end, s))
        
        # Display the menu at the cursor position
        menu.post(event.x_root, event.y_root)

# Function to replace a misspelled word with a suggestion
def replace_word(start, end, replacement):
    user_input_box.delete(start, end)  # Delete the misspelled word
    user_input_box.insert(start, replacement)  # Insert the suggested word
    check_spelling()  # Recheck spelling after replacement

def is_english(word):
    english_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return all(char in english_chars for char in word)

# Copy to Clipboard button
def copy_to_clipboard():
    root.clipboard_clear()  # Clear the clipboard
    root.clipboard_append(chat_history.get("1.0", tk.END))  # Copy chat history content to clipboard

def revise_last(event):
    global do_revise
    do_revise=True
    user_input_box.delete("1.0", tk.END)  # Clear current input
    user_input_box.insert(tk.END, previous_message)  # Insert previous message

def is_model_multimodal(model_name):
    MULTIMODAL_MODELS = {"bakllava", "llava", "cogvlm"}  # list of Multimodal LLMs for quick lookup
    try:
        model_name = model_name.lower()  # Normalize input
        # Check if model name starts with a known multimodal model name
        if any(model_name.startswith(model) for model in MULTIMODAL_MODELS):
            return True
        # Fallback: Query Ollama for metadata (if available)
        model_info = ollama.show(model_name)
        return model_info.get("details", {}).get("multimodal", False)
    except Exception as e:
        print(f"Error checking model capabilities: {e}")
        return False

# function to update the model description
def update_model_description():
    global do_send_images
    selected_model = model_var.get()
    multimodal = is_model_multimodal(selected_model)
    do_send_images = multimodal and do_read_image_var.get()

    if do_send_images:
        model_description_label.config(text="Multimodal model with image processing enabled", foreground="green")
    elif multimodal:
        model_description_label.config(text="Multimodal model detected (you can enable image reading)", foreground="orange")
    else:
        model_description_label.config(text="Standard text-based model", foreground="gray")

def on_toggle(var_name, *args):
    """Update global variables dynamically based on checkbox state."""
    globals()[var_name] = globals()[f"{var_name}_var"].get()
    # print(f"{var_name}: {globals()[var_name]}")  # Debugging output

# -------------------------------------------------

# Initialize spell checker
spell_checker = enchant.Dict("en_US")
previous_message=""
do_revise=False
do_send_images = False  # Initialize the boolean variable

# Create the main window
root = tk.Tk()
root.title("AI Document Analyzer")

# Set the background color of the main window to gray
root.configure(bg="#333333")

# Configure grid weights to make the chat history box resizable
root.grid_rowconfigure(1, weight=1)  # Make row 1 (chat history) resizable
root.grid_columnconfigure(0, weight=1)  # Make column 0 resizable

# Fetch installed Ollama models
installed_models = fetch_installed_models()

# Load user data (last folder path, last selected model, and temperature)
user_data = load_user_data()
last_folder = user_data.get("last_folder", "")
last_model = user_data.get("last_model", "")
last_temperature = user_data.get("temperature", 0.7)  # Default temperature
do_mention_page = user_data.get("do_mention_page", False)  # Default checkbox state
do_read_image = user_data.get("do_read_image", False)  # Default checkbox state

# Dropdown for model selection
model_var = tk.StringVar(root)
model_var.set(last_model if last_model in installed_models else (installed_models[0] if installed_models else "No models found"))
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=installed_models, state="readonly")
model_dropdown.grid(row=0, column=3, padx=10, pady=(0, 10), sticky="n")
model_dropdown.bind("<<ComboboxSelected>>", lambda event: update_model_description())

# Folder path entry
folder_path_entry = ttk.Entry(root, width=50)
folder_path_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

# Set the last used folder path
folder_path_entry.insert(0, last_folder)

# Folder path dropdown
folder_path_dropdown = ttk.Combobox(root, values=user_data.get("last_folders", []), state="normal")
folder_path_dropdown.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
folder_path_dropdown.bind("<<ComboboxSelected>>", on_folder_select)  # Bind selection event

# Set the dropdown value to the last_folder
folder_path_dropdown.set(last_folder)

# Browse button
browse_button = ttk.Button(root, text="Browse", command=browse_folder)  # Use the browse_folder function
browse_button.grid(row=0, column=1, padx=10, pady=10)

# Read button
read_button = ttk.Button(root, text="Read Documents", command=set_folder_path)
read_button.grid(row=0, column=2, padx=10, pady=10)

# Chat history display
chat_history = scrolledtext.ScrolledText(root, width=80, height=20, state=tk.DISABLED, bg="#444444", fg="white", insertbackground="white", wrap=tk.WORD)
chat_history.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

# Chat history display
chat_history = scrolledtext.ScrolledText(root, width=80, height=20, state=tk.DISABLED, bg="#444444", fg="white", insertbackground="white")
chat_history.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

# Sample text label
typehere_label = ttk.Label(root, text="Chat with AI here: (Shift+↵ to send | ^ to revise previous)", foreground="green",font=("Arial", 8))
typehere_label.grid(row=2, column=0, columnspan=4, padx=10, pady=(10, 0), sticky="w")  # Place between chat_history and user_input_box

# Configure tags for highlighting text
chat_history.tag_configure("fileread_tag", foreground="yellow")  # Color for file read report
chat_history.tag_configure("user_tag", foreground="cyan")  # Color for "You: "
chat_history.tag_configure("user_question", foreground="lightblue", font=("Arial", 10, "bold"))
chat_history.tag_configure("ai_tag", foreground="red")  # Color for "AI:"
chat_history.tag_configure("ai_response", foreground="white")  # Color for AI response text
chat_history.tag_configure("separator", foreground="gray")  # Color for the separator line

# User input box
user_input_box = tk.Text(root, width=70, height=5, bg="#444444", fg="white", insertbackground="white", wrap=tk.WORD)
user_input_box.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

# Configure the "misspelled" tag for underlining misspelled words
user_input_box.tag_configure("misspelled", underline=True, underlinefg="lightcoral")

# Bind the spell check function to the text widget
user_input_box.bind("<KeyRelease>", lambda event: check_spelling())

# Bind the right-click event to show spelling suggestions
user_input_box.bind("<Button-3>", show_suggestions)

# Send button
send_button = ttk.Button(root, text="Ask AI", command=chat_with_ai)
send_button.grid(row=3, column=1, padx=10, pady=10)

# Clear button
clear_button = tk.Button(root, text="Clear", command=clear_chat_history ,bg="lightcoral" ,fg="black")
clear_button.grid(row=4, column=1, padx=10, pady=10)


copy_button = ttk.Button(root, text="Copy History", command=copy_to_clipboard)
copy_button.grid(row=4, column=2, padx=10, pady=10)

# Temperature slider
temperature_label = ttk.Label(root, text="Innovation Factor:", background="#333333", foreground="white")
temperature_label.grid(row=3, column=2, padx=10, pady=10)

temperature_scale = tk.Scale(root, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, bg="#333333", fg="white", troughcolor="#444444")
temperature_scale.set(last_temperature)  # Set the last used temperature
temperature_scale.grid(row=3, column=3, padx=10, pady=10)

# Checkbox for toggling prompt
do_mention_var = tk.BooleanVar(value=do_mention_page)  # Set the checkbox state
do_mention_checkbox = ttk.Checkbutton(
    root,
    text="Tell Which Page",
    variable=do_mention_var,
    onvalue=True,
    offvalue=False
)
do_mention_checkbox.grid(row=4, column=3, padx=10, pady=10, sticky="w")

# Checkbox for toggling prompt
do_read_image_var = tk.BooleanVar(value=do_read_image)  # Set the checkbox state
do_read_image_checkbox = ttk.Checkbutton(
    root,
    text="Read Images",
    variable=do_read_image_var,
    onvalue=True,
    offvalue=False
)
do_read_image_checkbox.grid(row=5, column=3, padx=10, pady=10, sticky="w")

# Store the BooleanVar references in globals()
globals()["do_mention_page_var"] = do_mention_var
globals()["do_read_image_var"] = do_read_image_var

# Attach trace function
do_mention_var.trace_add("write", lambda *args: on_toggle("do_mention_page", *args))
do_read_image_var.trace_add("write", lambda *args: on_toggle("do_read_image", *args))


model_description_label = ttk.Label(root, text="Select a model", foreground="gray")
model_description_label.grid(row=0, column=3, padx=10, pady=(10, 0), sticky="s")

update_model_description()

# trace to update the label when either variable changes
model_var.trace_add("write", lambda *args: update_model_description())
# Modify the do_read_image_var trace to include model description updates
do_read_image_var.trace_add("write", lambda *args: (on_toggle("do_read_image", *args), update_model_description()))

# Bind the UP key to recall the previous user message
user_input_box.bind("<Up>", revise_last)

# Bind the Delete key to the folder_path_dropdown widget
folder_path_dropdown.bind("<Delete>", delete_folder_path)
# Bind the Enter key to trigger the chat_with_ai function
user_input_box.bind("<Shift-Return>", lambda event: chat_with_ai())

# Run the application
root.mainloop()