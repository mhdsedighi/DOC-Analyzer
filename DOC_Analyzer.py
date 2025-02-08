import os
import json
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton,
                             QTextEdit, QComboBox, QSlider, QCheckBox,
                             QMessageBox, QFileDialog, QMenu)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QTextCharFormat, QTextCursor, QSyntaxHighlighter, QColor
from PyQt6 import QtGui
from PyQt6.QtGui import QShortcut
import enchant
import ollama
from modules.file_read import extract_content_from_file
from modules.utils import load_user_data, save_user_data
from modules.custom_widgets import AddressMenu
import re, string, time, math

# Define the cache folder and ensure it exists
if not os.path.exists("cache"):
    os.makedirs("cache")

# File paths for saving user data and cache
USER_DATA_FILE = os.path.join("cache", "user_data.json")
DOCUMENT_CACHE_FILE = os.path.join("cache", "document_cache.json")

# Supported file extensions
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".pptx", ".ppt"]

chat_history_list = []  # List to store the conversation history

# Define color formats for chat history
user_format = QTextCharFormat()
user_format.setForeground(QColor("lightblue"))  # User messages in light blue

ai_format = QTextCharFormat()
ai_format.setForeground(QColor("lightgreen"))  # AI responses in light green

system_format = QTextCharFormat()
system_format.setForeground(QColor("gray"))  # System messages in gray

class OllamaWorkerThread(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    process_killed = pyqtSignal()  # Signal when the process is killed

    def __init__(self, model, messages, options):
        super().__init__()
        self.model = model
        self.messages = messages
        self.options = options
        self.is_running = True  # Flag to control the loop

    def run(self):
        try:
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                options=self.options
            )
            if self.is_running:
                ai_response = response['message']['content']
                self.response_received.emit(ai_response)
        except Exception as e:
            if self.is_running:  # Check before emitting the error signal
                self.error_occurred.emit(str(e))

    def stop(self):
        self.is_running = False  # Set the flag to exit the loop
        self.process_killed.emit()  # Emit the signal after attempting to sto


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
        QMessageBox.critical(window, "Error", f"Failed to fetch models: {e}")
        return []


# Function to handle folder browsing
def browse_folder():
    folder_path = QFileDialog.getExistingDirectory(window, "Select folder with any number of PDF, DOCX, TXT ,PPT, ... files")
    if folder_path:
        address_menu.set_current_address(folder_path)  # Set the selected folder
        address_menu.add_address(folder_path)  # Add to the list if not already present
        save_user_data(last_folder=folder_path)  # Save the last folder

# Function to read all documents in a folder
def read_documents(folder_path):
    document_text = []  # this appears to fix adding extra texts when changing folder
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
    if do_mention_page_var.isChecked():
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
                if do_read_image and not image_content:  # and not file_ext=="txt"
                    text_content, image_content, word_count, readable_percentage = extract_content_from_file(file_path, do_read_image)

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

            chat_history.append(f"Looked at: {filename}\n")
            chat_history.append(f"Word count: {word_count}\n")
            chat_history.append(f"Extracted images: {len(image_content)}\n")
            chat_history.append(f"Readable content: {readable_percentage:.2f}%\n")


            # Add images to document_images list if model supports images
            if do_send_images:
                document_images.extend(image_content)
    enable_ai_interaction()
    return all_text, new_files_read, document_images

# Function to handle the chat with the AI
def chat_with_ai():
    user_input = user_input_box.toPlainText().strip()
    global previous_message
    global do_revise
    previous_message = user_input

    if do_revise:  # Removing what has been revised from prompt
        if chat_history_list:
            chat_history_list.pop()
            chat_history_list.pop()
        do_revise = False

    if user_input:
        # Append user message with user format
        cursor = chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)  # Move cursor to the end
        cursor.insertText("You: ", user_format)  # Insert "You: " with user format
        cursor.insertText(f"{user_input}\n", user_format)  # Insert user input with user format
        chat_history.setTextCursor(cursor)  # Update the cursor position
        chat_history.ensureCursorVisible()  # Scroll to the bottom

        # Add the user's message to the chat history list
        chat_history_list.append({"role": "user", "content": user_input})

        # Combine the document text with the chat history
        full_prompt = f"{document_text}\n\n" + "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history_list])

        try:
            # Send the prompt to the AI using the selected model
            selected_model = model_var.currentText()
            messages = [{"role": "user", "content": full_prompt}]

            # If the model supports images, include them in the messages
            if do_send_images:
                for img_base64 in document_images:
                    messages.append({"role": "user", "content": f"data:image/png;base64,{img_base64}"})

            # Change the button to "Kill Process"
            send_button.setText("Cancel")
            send_button.setStyleSheet("background-color: red; color: white;")
            send_button.clicked.disconnect()  # Disconnect the previous connection
            send_button.clicked.connect(kill_ollama_process)  # Connect to kill function
            user_input_box.setEnabled(False)

            # Create and start the worker thread
            global ollama_worker
            ollama_worker = OllamaWorkerThread(selected_model, messages, {"temperature": temperature_scale.value() / 100.0})
            ollama_worker.response_received.connect(handle_ai_response)
            ollama_worker.error_occurred.connect(handle_ai_error)
            ollama_worker.process_killed.connect(reset_ask_ai_button)
            ollama_worker.start()
        except Exception as e:
            QMessageBox.critical(window, "Error", f"An error occurred: {e}")

        user_input_box.clear()
        chat_history.verticalScrollBar().setValue(chat_history.verticalScrollBar().maximum())  # Scroll down
        # Save the last used model, folder path, temperature, and checkbox state
        save_user_data(
            address_menu.get_current_address().strip(),
            selected_model,
            temperature_scale.value() / 100.0,
            do_mention_page=do_mention_page,
            do_read_image=do_read_image
        )

# Function to handle AI response
def handle_ai_response(response):
    # Append AI response with AI format
    cursor = chat_history.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)  # Move cursor to the end
    cursor.insertText("A.I.: ", ai_format)  # Insert "AI: " with AI format
    cursor.insertText(f"{response}\n", ai_format)  # Insert AI response with AI format
    chat_history.setTextCursor(cursor)  # Update the cursor position
    chat_history.ensureCursorVisible()  # Scroll to the bottom

    # Add a horizontal line after the AI's response
    cursor = chat_history.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)  # Move cursor to the end
    cursor.insertText("---\n", system_format)  # Insert separator with system format
    chat_history.setTextCursor(cursor)  # Update the cursor position
    chat_history.ensureCursorVisible()  # Scroll to the bottom

    # Add the AI's response to the chat history list
    chat_history_list.append({"role": "assistant", "content": response})

    # Reset the button to "Ask AI"
    reset_ask_ai_button()

# Function to handle AI errors
def handle_ai_error(error_message):
    QMessageBox.critical(window, "Error", f"An error occurred: {error_message}")
    reset_ask_ai_button()


# Function to reset the "Ask AI" button
def reset_ask_ai_button():
    send_button.setText("Ask AI")
    send_button.setStyleSheet("""
        QPushButton {
            border: 2px solid blue;
            border-radius: 5px;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: rgba(0, 0, 255, 0.3);
        }
        QPushButton:disabled {
            color: darkgray;
            background-color: lightgray;
            border-color: gray;
        }
    """)
    send_button.clicked.disconnect()  # Disconnect the kill function
    send_button.clicked.connect(chat_with_ai)  # Reconnect to the chat function
    user_input_box.setEnabled(True)

# Function to kill the Ollama process
def kill_ollama_process():
    # Show a confirmation dialog
    confirm = QMessageBox.question(
        window,  # Parent window
        "Cancel Process",  # Dialog title
        "Are you sure you abort the processing of your question?",  # Dialog message
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,  # Buttons
        QMessageBox.StandardButton.No  # Default button
    )

    # If the user confirms, kill the process
    if confirm == QMessageBox.StandardButton.Yes:
        if hasattr(ollama_worker, "stop"):
            ollama_worker.stop()
        reset_ask_ai_button()


# Function to clear the chat history box
def clear_chat_history():
    chat_history.clear()
    chat_history_list.clear()  # Clear the chat history list


# Function to set the folder path
def set_folder_path():
    global document_text
    document_text = ""  # Clear the previous document text
    global document_images
    read_button.setText("Reading...")
    QApplication.processEvents()  # Force the UI update if you don't use QThread
    document_images = []
    folder_path = address_menu.get_current_address()  # Get the current address
    if not folder_path:
        QMessageBox.warning(window, "Warning", "Please enter a folder path.")
        return

    if not os.path.isdir(folder_path):
        QMessageBox.critical(window, "Error", "Invalid folder path.")
        return

    disable_ai_interaction()
    # Read documents from the new folder
    document_text, new_files_read, document_images = read_documents(folder_path)
    cursor = chat_history.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)  # Move cursor to the end
    cursor.insertText(f"Documents reading finished. {new_files_read} items were new for the library.\n", system_format)
    cursor.insertText("You can now chat with the A.I.\n", system_format)
    chat_history.setTextCursor(cursor)  # Update the cursor position
    chat_history.ensureCursorVisible()  # Scroll to the bottom

    # Save the folder path, last used model, and temperature
    save_user_data(folder_path, model_var.currentText(), temperature_scale.value() / 100.0)
    enable_ai_interaction()

class SpellCheckTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.highlighter = SpellCheckHighlighter(self.document())  # Attach highlighter

        # Enable custom context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_spell_menu)

    def show_spell_menu(self, pos):
        cursor = self.cursorForPosition(pos)
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        word = cursor.selectedText()

        if not word or self.highlighter.is_valid_word(word):
            return  # Exit if no word or it's valid

        menu = QMenu(self)
        suggestions = self.highlighter.spell_checker.suggest(word)

        if suggestions:
            for suggestion in suggestions[:5]:  # Show up to 5 suggestions
                action = menu.addAction(suggestion)
                action.triggered.connect(lambda checked, s=suggestion: self.replace_word(cursor, s))

        menu.exec(self.mapToGlobal(pos))

    def replace_word(self, cursor, new_word):
        cursor.insertText(new_word)

class SpellCheckHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)
        self.spell_checker = enchant.Dict("en_US")  # Use English dictionary

        # Define the red wavy underline format
        self.error_format = QTextCharFormat()
        self.error_format.setUnderlineColor(QColor("red"))
        self.error_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)

    def highlightBlock(self, text):
        words = re.findall(r'\b\w+\b', text)  # Extract words
        for word in words:
            if not self.is_valid_word(word):
                start = text.index(word)
                self.setFormat(start, len(word), self.error_format)

    def is_valid_word(self, word): # Ensure the word contains only English letters (A-Z, a-z)
        if not all(char in string.ascii_letters for char in word):
            return True  # Ignore words with non-English characters (consider them valid)
        return self.spell_checker.check(word)


# Copy to Clipboard button
def copy_to_clipboard():
    clipboard = QApplication.clipboard()
    clipboard.setText(chat_history.toPlainText())


def revise_last():
    global do_revise
    do_revise = True
    user_input_box.clear()
    user_input_box.insertPlainText(previous_message)


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
    selected_model = model_var.currentText()
    multimodal = is_model_multimodal(selected_model)
    do_send_images = multimodal and do_read_image_var.isChecked()

    if do_send_images:
        model_description_label.setText("Multimodal model with image processing enabled")
        model_description_label.setStyleSheet("color: green;")
    elif multimodal:
        model_description_label.setText("Multimodal model detected (you can enable image reading)")
        model_description_label.setStyleSheet("color: orange;")
    else:
        model_description_label.setText("Standard text-based model")
        model_description_label.setStyleSheet("color: gray;")


def on_toggle(var_name):
    """Update global variables dynamically based on checkbox state."""
    globals()[var_name] = globals()[f"{var_name}_var"].isChecked() # Access the QCheckBox variable
    # print(f"{var_name}: {globals()[var_name]}")  # Debugging output


# -------------------------------------------------



previous_message = ""
do_revise = False
do_send_images = False  # Initialize the boolean variable
ollama_worker = None

# Create the main window
app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("AI Document Analyzer")

# Set the background color of the main window to gray
window.setStyleSheet("background-color: #333333;")

# Create a central widget
central_widget = QWidget()
window.setCentralWidget(central_widget)

# Create a main layout (vertical)
main_layout = QVBoxLayout()
central_widget.setLayout(main_layout)

# Create a horizontal layout for the top section
top_layout = QHBoxLayout()
main_layout.addLayout(top_layout)

# Fetch installed Ollama models
installed_models = fetch_installed_models()


read_button_timer = QTimer()
read_button_timer.timeout.connect(lambda: animate_button(read_button))

# Load user data (last folder path, last selected model, and temperature)
user_data = load_user_data()
last_folder = user_data.get("last_folder", "")
last_model = user_data.get("last_model", "")
last_temperature = user_data.get("temperature", 0.7)  # Default temperature
do_mention_page = user_data.get("do_mention_page", False)  # Default checkbox state
do_read_image = user_data.get("do_read_image", False)  # Default checkbox state

# Dropdown for model selection
model_var = QComboBox()
model_var.addItems(installed_models)
model_var.setCurrentText(last_model if last_model in installed_models else (installed_models[0] if installed_models else "No models found"))
top_layout.addWidget(model_var)
model_var.currentIndexChanged.connect(update_model_description)


# Adding the custom AddressMenu
address_menu = AddressMenu()
top_layout.addWidget(address_menu)

# Browse button
browse_button = QPushButton("Browse")
browse_button.clicked.connect(browse_folder)
top_layout.addWidget(browse_button)

# Read button
read_button = QPushButton("Read Documents")
read_button.clicked.connect(set_folder_path)
#read_button.clicked.connect(lambda: (read_button.setText("Reading..."),set_folder_path()))
top_layout.addWidget(read_button)

# Chat history display
chat_history = QTextEdit()
chat_history.setReadOnly(True)
chat_history.setStyleSheet("background-color: #444444; color: white;")
main_layout.addWidget(chat_history)

# Sample text label
typehere_label = QLabel("Chat with A.I. here: (Ctrl+↵ to send | Ctrl+^ to revise previous)")
typehere_label.setStyleSheet("color: green;")
main_layout.addWidget(typehere_label)

# User input box
user_input_box = SpellCheckTextEdit()
user_input_box.setStyleSheet("background-color: #444444; color: white;")
user_input_box.setMaximumHeight(100)
user_input_box.setPlaceholderText("Please read documents first.")  # Set placeholder text
main_layout.addWidget(user_input_box)


# Create a horizontal layout for the buttons and slider
bottom_layout = QHBoxLayout()
main_layout.addLayout(bottom_layout)

# Send button
send_button = QPushButton("Ask A.I.")
send_button.setStyleSheet("""
    QPushButton {
        border: 2px solid blue;
        border-radius: 5px;
        padding: 5px 10px;
    }
    QPushButton:hover {
        background-color: rgba(0, 0, 255, 0.3);
    }
    QPushButton:disabled {  /* More specific */
        color: darkgray;
        background-color: lightgray; /* Example disabled background */
        border-color: gray;       /* Example disabled border */
    }
""")

send_button.clicked.connect(chat_with_ai)
send_button.setEnabled(False)  # Disable initially

# Clear button
clear_button = QPushButton("Clear")
clear_button.setStyleSheet("background-color: lightcoral; color: black;")
clear_button.clicked.connect(clear_chat_history)


copy_button = QPushButton("Copy History")
copy_button.clicked.connect(copy_to_clipboard)

# Temperature slider
temperature_label = QLabel("Innovation Factor:")
temperature_label.setStyleSheet("color: white;")


# Function to update the temperature label and save the value
def update_temperature_label(value):
    temperature_value = value / 10.0  # Convert slider value to 0.1 increments
    temperature_display_label.setText(f"{temperature_value:.1f}")
    save_user_data(temperature=temperature_value)  # Save the temperature value to cache

def disable_ai_interaction():
    """Disables AI interaction elements (Ask AI button and user input box)."""
    send_button.setEnabled(False)
    user_input_box.setEnabled(False)
    user_input_box.setPlaceholderText("Please read documents first.")
    """Starts the blinking animation on the Read Documents button."""
    read_button_timer.start(500)

def enable_ai_interaction():
    """Enables AI interaction elements."""
    read_button.setText("Read Documents")
    send_button.setEnabled(True)
    user_input_box.setEnabled(True)
    user_input_box.setPlaceholderText("")  # Remove placeholder text
    """Stops the blinking animation on the Read Documents button."""
    read_button_timer.stop()
    read_button.setStyleSheet("") # Reset style
    read_button.setText("Read Documents")

def animate_button(button):
    alpha = int((1 +  math.sin(time.time() * 2)) * 127.5)  # Sinusoidal alpha
    color = QColor(255, 255, 255, alpha) # White with varying alpha
    button.setStyleSheet(f"background-color: rgba({color.red()},{color.green()},{color.blue()},{color.alpha()});")

# Label to display the current temperature value
temperature_display_label = QLabel(f"{last_temperature:.1f}")
temperature_display_label.setStyleSheet("color: white;")

# Temperature slider with 0.1 increments
temperature_scale = QSlider(Qt.Orientation.Horizontal)
temperature_scale.setMinimum(0)
temperature_scale.setMaximum(10)  # Represents 0.0 to 1.0 in 0.1 increments
temperature_scale.setValue(int(last_temperature * 10))  # Set the last used temperature
temperature_scale.valueChanged.connect(update_temperature_label)  # Update label and save value
temperature_scale.setMaximumWidth(100)

# add to layout
bottom_layout.setSpacing(10)
bottom_layout.addWidget(clear_button)
bottom_layout.addWidget(copy_button)
bottom_layout.addWidget(temperature_label)
bottom_layout.addWidget(temperature_scale)
bottom_layout.addWidget(temperature_display_label)
bottom_layout.addWidget(send_button)

# Checkbox for toggling prompt
do_mention_page_var = QCheckBox("Tell Which Page")  # Keep the variable name consistent
do_mention_page_var.setChecked(do_mention_page)
do_mention_page_var.stateChanged.connect(lambda: on_toggle("do_mention_page"))  # Pass the string name
bottom_layout.addWidget(do_mention_page_var)
do_mention_page = do_mention_page_var.isChecked()  # Initialize the boolean variable

# Checkbox for toggling image reading
do_read_image_var = QCheckBox("Read Images")
do_read_image_var.setChecked(do_read_image)
do_read_image_var.stateChanged.connect(lambda: (on_toggle("do_read_image"), update_model_description()))
do_read_image = do_read_image_var.isChecked()  # Initialize the boolean variable

top_layout.addWidget(do_read_image_var)

model_description_label = QLabel("Select a model")
model_description_label.setStyleSheet("color: gray;")
top_layout.addWidget(model_description_label)

update_model_description()

model_var.currentIndexChanged.connect(update_model_description)
address_menu.folder_changed.connect(disable_ai_interaction)

# Bind the UP key to recall the previous user message

user_input_box.shortcut = QShortcut(QtGui.QKeySequence("Ctrl+Up"), user_input_box)
user_input_box.shortcut.activated.connect(revise_last)

# Bind the Enter key to trigger the chat_with_ai function
user_input_box.shortcut = QShortcut(QtGui.QKeySequence("Ctrl+Return"), user_input_box)
user_input_box.shortcut.activated.connect(chat_with_ai)

# Set window size and position (optional - adjust as needed)
window.setGeometry(100, 100, 800, 600)  # Example size
disable_ai_interaction() # for initial app start
window.show()
# apply style to all disabled buttons in the window:
sys.exit(app.exec())