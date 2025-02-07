import os
import json
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTextEdit, QComboBox, QSlider, QCheckBox,
                             QMessageBox, QFileDialog, QSizePolicy,QMenu,QStyledItemDelegate)
from PyQt6.QtCore import Qt, QSize, QRect
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QTextCursor,QSyntaxHighlighter,QBrush,QColor
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtGui import QShortcut, QPainter
import enchant
import ollama
from modules.file_read import extract_content_from_file
import base64
import re
import string

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
        QMessageBox.critical(window, "Error", f"Failed to fetch models: {e}")
        return []


# Function to handle folder browsing
def browse_folder():
    folder_path = QFileDialog.getExistingDirectory(window, "Select Folder")
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
            chat_history.append(f"Extracted images: {len(image_content)}\n\n")
            chat_history.append(f"Readable content: {readable_percentage:.2f}%\n\n")


            # Add images to document_images list if model supports images
            if do_send_images:
                document_images.extend(image_content)

    return all_text, new_files_read, document_images

# Custom delegate to add a button inside each combo box item
class RemoveButtonDelegate(QStyledItemDelegate):
    def __init__(self, parent, remove_callback):
        super().__init__(parent)
        self.remove_callback = remove_callback  # Function to remove items

    def paint(self, painter, option, index):
        """Draw the item text and a small remove button."""
        super().paint(painter, option, index)
        painter.save()

        # Define button area (small red cross on the right)
        button_rect = option.rect.adjusted(option.rect.width() - 20, 2, -2, -2)
        painter.setPen(Qt.GlobalColor.red)
        painter.drawText(button_rect, Qt.AlignmentFlag.AlignCenter, "×")

        painter.restore()

    def editorEvent(self, event, model, option, index):
        """Handle button clicks inside the combo box."""
        if event.type() == event.Type.MouseButtonPress:
            button_rect = option.rect.adjusted(option.rect.width() - 20, 2, -2, -2)
            if button_rect.contains(event.pos()):
                self.remove_callback(index.row())  # Call the remove function
                return True
        return False

# Custom address menu where users can type in or select addresses
class AddressMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_data = load_user_data()  # Load stored addresses
        self.init_ui()

    def init_ui(self):
        # Main layout for the AddressMenu
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # ComboBox for selecting addresses
        self.combo_box = QComboBox()
        self.combo_box.setEditable(True)  # Allow typing in the combo box
        self.combo_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)  # Prevent auto-insert
        self.combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_box.addItems(self.user_data.get("last_folders", []))  # Load initial addresses
        self.combo_box.setCurrentText(self.user_data.get("last_folder", ""))  # Set current address
        layout.addWidget(self.combo_box)

        # Assign custom delegate to handle remove buttons
        self.combo_box.setItemDelegate(RemoveButtonDelegate(self.combo_box, self.remove_item))

    def remove_item(self, index):
        """Remove an address from the combo box and update user data."""
        if 0 <= index < self.combo_box.count():
            self.combo_box.removeItem(index)
            self.save_addresses()  # Update stored addresses

    def save_addresses(self):
        """Save the current list of addresses to user data, ensuring empty lists are handled."""
        addresses = [self.combo_box.itemText(i) for i in range(self.combo_box.count())]
        save_user_data(last_folders=addresses, last_folder=self.combo_box.currentText().strip())  # Persist updated addresses

    def get_current_address(self):
        """Get the currently selected or typed address."""
        return self.combo_box.currentText().strip()

    def set_current_address(self, address):
        """Set the current address in the combo box."""
        self.combo_box.setCurrentText(address)

    def add_address(self, address):
        """Add a new address to the combo box if it doesn’t already exist."""
        if address and self.combo_box.findText(address) == -1:  # Avoid duplicates
            self.combo_box.insertItem(0, address)  # Add to the top
            self.combo_box.setCurrentIndex(0)  # Set as current
            self.save_addresses()  # Update user data

# Function to handle the chat with the AI
def chat_with_ai():
    user_input = user_input_box.toPlainText().strip()
    global previous_message
    global do_revise
    previous_message=user_input

    if do_revise:  #removing what has been revised from prompt
        if chat_history_list:
            chat_history_list.pop()
            chat_history_list.pop()
        do_revise=False

    if user_input:
        chat_history.append("You: ")  # Tag for user text
        chat_history.append(f"{user_input}\n")  # Tag for user question

        # Add the user's message to the chat history list
        chat_history_list.append({"role": "user", "content": user_input})

        # Combine the document text with the chat history
        full_prompt = f"{document_text}\n\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_list])

        try:
            # Send the prompt to the AI using the selected model
            selected_model = model_var.currentText()
            messages = [{"role": "user", "content": full_prompt}]

            # If the model supports images, include them in the messages
            if do_send_images:
                for img_base64 in document_images:
                    messages.append({"role": "user", "content": f"data:image/png;base64,{img_base64}"})

            response = ollama.chat(
                model=selected_model,
                messages=messages,
                options={"temperature": temperature_scale.value() / 100.0}  # Use the slider value
            )

            ai_response = response['message']['content']
            chat_history.append("AI: ")  # Tag for "AI:"
            chat_history.append(f"{ai_response}\n")  # Tag for AI response

            # Add a horizontal line after the AI's response
            chat_history.append("---\n")  # Tag for the separator line

            # Add the AI's response to the chat history list
            chat_history_list.append({"role": "assistant", "content": ai_response})
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


# Function to clear the chat history box
def clear_chat_history():
    chat_history.clear()
    chat_history_list.clear()  # Clear the chat history list


# Function to set the folder path
def set_folder_path():
    global document_text
    document_text = ""  # Clear the previous document text
    global document_images
    document_images = []

    folder_path = address_menu.get_current_address()  # Get the current address
    if not folder_path:
        QMessageBox.warning(window, "Warning", "Please enter a folder path.")
        return

    if not os.path.isdir(folder_path):
        QMessageBox.critical(window, "Error", "Invalid folder path.")
        return

    # Read documents from the new folder
    document_text, new_files_read, document_images = read_documents(folder_path)
    chat_history.append(f"Documents reading finished. {new_files_read} new files were processed.\n")
    chat_history.append("You can now chat with the A.I.\n")
    chat_history.verticalScrollBar().setValue(chat_history.verticalScrollBar().maximum())  # Scroll down

    # Save the folder path, last used model, and temperature
    save_user_data(folder_path, model_var.currentText(), temperature_scale.value() / 100.0)

    # Read documents from the new folder
    document_text, new_files_read, document_images = read_documents(folder_path)
    chat_history.append(f"Documents reading finished. {new_files_read} new files were processed.\n")
    chat_history.append("You can now chat with the A.I.\n")
    chat_history.verticalScrollBar().setValue(chat_history.verticalScrollBar().maximum())  # scrolling down

    # Save the folder path, last used model, and temperature
    save_user_data(folder_path, model_var.currentText(), temperature_scale.value() / 100.0)

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


# Adding the AddressMenu
address_menu = AddressMenu()
top_layout.addWidget(address_menu)

# Browse button
browse_button = QPushButton("Browse")
browse_button.clicked.connect(browse_folder)
top_layout.addWidget(browse_button)

# Read button
read_button = QPushButton("Read Documents")
read_button.clicked.connect(set_folder_path)
top_layout.addWidget(read_button)

# Chat history display
chat_history = QTextEdit()
chat_history.setReadOnly(True)
chat_history.setStyleSheet("background-color: #444444; color: white;")
main_layout.addWidget(chat_history)

# Sample text label
typehere_label = QLabel("Chat with AI here: (Shift+↵ to send | ^ to revise previous)")
typehere_label.setStyleSheet("color: green;")
main_layout.addWidget(typehere_label)

# User input box
user_input_box = SpellCheckTextEdit()
user_input_box.setStyleSheet("background-color: #444444; color: white;")
user_input_box.setMaximumHeight(100)
main_layout.addWidget(user_input_box)


# Create a horizontal layout for the buttons and slider
bottom_layout = QHBoxLayout()
main_layout.addLayout(bottom_layout)

# Send button
send_button = QPushButton("Ask AI")
send_button.setStyleSheet("""
    QPushButton {
        border: 2px solid blue;  /* Blue border */
        border-radius: 5px;  /*  rounded corners */
        padding: 5px 10px;  /* Padding for better appearance */
    }
    QPushButton:hover {
        background-color: rgba(0, 0, 255, 0.3);  /* Light blue on hover */
    }
""")

send_button.clicked.connect(chat_with_ai)


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
do_mention_page_var.stateChanged.connect(lambda: on_toggle("do_mention_page")) # Pass the string name
bottom_layout.addWidget(do_mention_page_var)
do_mention_page = do_mention_page_var.isChecked() # Initialize the boolean variable

# Checkbox for toggling image reading
do_read_image_var = QCheckBox("Read Images")
do_read_image_var.setChecked(do_read_image)
do_read_image_var.stateChanged.connect(lambda: (on_toggle("do_read_image"),update_model_description()))
do_read_image = do_read_image_var.isChecked() # Initialize the boolean variable

top_layout.addWidget(do_read_image_var)

model_description_label = QLabel("Select a model")
model_description_label.setStyleSheet("color: gray;")
top_layout.addWidget(model_description_label)

update_model_description()

model_var.currentIndexChanged.connect(update_model_description)

# Bind the UP key to recall the previous user message
user_input_box.shortcut = QShortcut(QtGui.QKeySequence("Up"), user_input_box)
user_input_box.shortcut.activated.connect(revise_last)


# Bind the Enter key to trigger the chat_with_ai function
user_input_box.shortcut = QShortcut(QtGui.QKeySequence("Shift+Return"), user_input_box)
user_input_box.shortcut.activated.connect(chat_with_ai)

# Set window size and position (optional - adjust as needed)
window.setGeometry(100, 100, 800, 600)  # Example size

window.show()
sys.exit(app.exec())