from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSlider, QCheckBox, QMessageBox, QFileDialog, QMenu,
    QDialog, QLineEdit, QSpacerItem, QSizePolicy, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QTextCharFormat, QTextCursor, QSyntaxHighlighter, QColor, QFont, QFontDatabase, QShortcut
from PyQt6 import QtGui
import ollama, enchant, sys, os, json, re, string, time, math, psutil, pynvml
from modules.file_read import extract_content_from_file, is_file_in_chroma, set_chroma_db
from modules.utils import load_user_data, save_user_data
from modules.custom_widgets import AddressMenu
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.memory import BaseMemory
import chromadb
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the cache folder and ensure it exists
if not os.path.exists("cache"):
    os.makedirs("cache")
CHROMA_DIR = os.path.join("cache", "chroma_db_docs")
if not os.path.exists(CHROMA_DIR):
    os.makedirs(CHROMA_DIR)
CHROMA_MEM_DIR = os.path.join("cache", "chroma_db_mem")  # New memory database
if not os.path.exists(CHROMA_MEM_DIR):
    os.makedirs(CHROMA_MEM_DIR)

# File paths for saving user data and cache
USER_DATA_FILE = os.path.join("cache", "user_data.json")
DOCUMENT_CACHE_FILE = os.path.join("cache", "document_cache.json")

# Supported file extensions
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".pptx", ".ppt"]

chat_history_list = []  # List to store the conversation history

# Define color formats for chat history
user_format = QTextCharFormat()
user_format.setForeground(QColor("blue"))  # User messages in light blue

ai_format = QTextCharFormat()
# ai_format.setForeground(QColor("lightgreen"))  # AI responses in light green

system_format = QTextCharFormat()
system_format.setForeground(QColor("gray"))  # System messages in gray


# Custom memory class inspired by second codebase
class CustomChatMemory(BaseMemory):
    chat_memory: InMemoryChatMessageHistory = InMemoryChatMessageHistory()
    ai_prefix: str = "AI"

    def __init__(self, chat_memory=None, ai_prefix="AI"):
        super().__init__()
        self.chat_memory = chat_memory or InMemoryChatMessageHistory()
        self.ai_prefix = ai_prefix

    def load_memory_variables(self, inputs):
        messages = self.chat_memory.messages
        return {"context": "\n".join([msg.content for msg in messages])}

    def save_context(self, inputs, outputs):
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")
        if input_str:
            self.chat_memory.add_user_message(input_str)
        if output_str:
            self.chat_memory.add_ai_message(output_str)

    def clear(self):
        self.chat_memory.clear()

    @property
    def memory_variables(self):
        return ["context"]


class OllamaWorkerThread(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    process_killed = pyqtSignal()  # Signal when the process is killed

    def __init__(self, model, messages, options, chain, memory, retriever_docs, retriever_mem, document_images, do_send_images):
        super().__init__()
        self.model = model
        self.messages = messages
        self.options = options
        self.chain = chain
        self.memory = memory
        self.retriever_docs = retriever_docs
        self.retriever_mem = retriever_mem
        self.document_images = document_images
        self.do_send_images = do_send_images
        self.is_running = True  # Flag to control the loop

    def orchestrate_retrievers(self, query: str):
        # Fetch memory context with date relevance
        mem_result = (
                self.retriever_mem
                | self.add_date_to_page_content
                | self.filter_by_date_relevance
        ).invoke(query)

        # Fetch document context
        doc_result = self.retriever_docs.invoke(query)

        # Combine results
        result = mem_result + doc_result

        # Log final retrieval
        for d in result:
            logger.info(
                f"Retrieved: Title: {d.metadata.get('filename', 'N/A')}, Author: {d.metadata.get('author', 'N/A')}, Content: {d.page_content}")

        return "\n".join([d.page_content for d in result])

    def add_date_to_page_content(self, docs):
        for d in docs:
            timestamp = d.metadata.get("timestamp", datetime.today().strftime('%Y-%m-%d'))
            days_ago = (datetime.today() - datetime.strptime(timestamp, '%Y-%m-%d')).days
            time_ago = f"{days_ago} days ago" if days_ago > 0 else "Today"
            d.page_content = f"[{time_ago}] {d.page_content}"
        return docs

    def filter_by_date_relevance(self, docs, max_len=3, recent_amt=2, older_amt=1):
        if len(docs) <= max_len:
            return docs
        updated_docs = []
        docs_tuples = [
            (datetime.strptime(d.metadata.get("timestamp", datetime.today().strftime('%Y-%m-%d')), '%Y-%m-%d'), d) for d
            in docs]
        docs_tuples_sorted = sorted(docs_tuples, key=lambda dtup: dtup[0], reverse=True)
        for i in range(min(recent_amt, len(docs_tuples_sorted))):
            updated_docs.append(docs_tuples_sorted[i][1])
        docs_tuples_sorted = docs_tuples_sorted[recent_amt:]
        from random import randrange
        for i in range(min(older_amt, len(docs_tuples_sorted))):
            rnd_index = randrange(len(docs_tuples_sorted))
            updated_docs.append(docs_tuples_sorted[rnd_index][1])
            del docs_tuples_sorted[rnd_index]
        return updated_docs

    def run(self):
        try:
            # Build LangChain pipeline within the thread
            prompt = PromptTemplate(
                input_variables=["context", "input"],
                template="""You are an AI assistant analyzing documents and conversation history.
                Provide detailed responses based on the context. If unsure, say 'I don’t know'.
                Context: {context}
                User: {input}
                AI:"""
            )
            chain = (
                    {
                        "context": lambda x: self.orchestrate_retrievers(x),
                        "input": RunnablePassthrough()
                    }
                    | prompt
                    | ChatOllama(model=self.model, temperature=self.options["temperature"])
                    | StrOutputParser()
            )

            # Prepare full input
            full_input = json.dumps(self.messages, indent=2)
            response = chain.invoke(full_input)

            # Optional web search if response is uncertain
            if len(response) < 200 and "don’t know" in response.lower() and WEB_SEARCH_ENABLED:
                search_context = self.get_web_search(full_input)
                response = chain.invoke(f"{full_input}\nWeb Search Context: {search_context}")

            if self.is_running:
                self.response_received.emit(response)

                # Save to memory
                self.memory.save_context({"input": full_input}, {"output": response})
                summary = chain.invoke(f"Summarize in 10 words or less: '{full_input}'")
                chroma_db_mem.add_texts(
                    texts=[summary],
                    metadatas=[{"timestamp": datetime.today().strftime('%Y-%m-%d')}]
                )
        except Exception as e:
            if self.is_running:
                self.error_occurred.emit(str(e))

    def get_web_search(self, query, result_count=3):
        try:
            search_query = re.sub(r'[^a-zA-Z0-9\s]', '', query)[:50]  # Simplify query
            search_results = subprocess.check_output(
                f"ddgr -n {result_count} -r ie-en -C --unsafe --np \"{search_query}\"",
                shell=True
            ).decode("utf-8")
            return f"Web search results:\n{search_results}"
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "Web search unavailable."

    def stop(self):
        self.is_running = False  # Set the flag to exit the loop
        self.process_killed.emit()  # Emit the signal after attempting to sto


# Load the document cache
def load_document_cache():
    if os.path.exists(DOCUMENT_CACHE_FILE):
        with open(DOCUMENT_CACHE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


# Save the document cache
def save_document_cache(cache):
    with open(DOCUMENT_CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(cache, file, indent=4, ensure_ascii=False)


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
    folder_path = QFileDialog.getExistingDirectory(window,
                                                   "Select folder with any number of PDF, DOCX, TXT ,PPT, ... files")
    if folder_path:
        address_menu.set_current_address(folder_path)  # Set the selected folder
        address_menu.add_address(folder_path)  # Add to the list if not already present
        save_user_data(last_folder=folder_path)  # Save the last folder


# Initialize Chroma client for document retrieval
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,  # Explicitly set the path
    settings=chromadb.Settings(
        is_persistent=True,
        persist_directory=CHROMA_DIR,
        allow_reset=True
    )
)
chroma_db = Chroma(
    client=chroma_client,
    collection_name="documents",
    embedding_function=FastEmbedEmbeddings()
)
logger.info(f"Chroma initialized with persist_directory: {chroma_client._system.settings.persist_directory}")
set_chroma_db(chroma_db)  # Pass the Chroma instance to file_read.py

# Initialize Chroma client for memory
chroma_mem_client = chromadb.PersistentClient(
    path=CHROMA_MEM_DIR,
    settings=chromadb.Settings(
        is_persistent=True,
        persist_directory=CHROMA_MEM_DIR,
        allow_reset=True
    )
)
chroma_db_mem = Chroma(
    client=chroma_mem_client,
    collection_name="mem",
    embedding_function=FastEmbedEmbeddings()
)


# Function to read all documents in a folder
def read_documents(folder_path):
    global document_text  # this appears to fix adding extra texts when changing folder
    cache = load_document_cache()
    new_files_read = 0  # Track the number of new files read
    document_images = []  # List to store images from documents

    # Set the introductory line based on the checkbox state. 4 Scenarios
    if not do_send_images:
        all_text = """You are analyzing extracted text from multiple documents stored in a Chroma vector database. Provide insights, summaries, or answers based on the retrieved textual content."""
    else:
        all_text = """You are analyzing extracted text and images from multiple documents stored in a Chroma vector database. The text may contain placeholders like [IMAGE_1], [IMAGE_2], etc., corresponding to images provided separately as base64 strings. Consider both text and these visual elements when generating insights, summaries, or answers."""

    if do_mention_page_var.isChecked():
        all_text += """ When providing insights, summaries, or answers, reference the file name and page number where the information was found."""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in SUPPORTED_EXTENSIONS:
            last_modified = os.path.getmtime(file_path)

            # Check if file needs to be re-read (not in Chroma or modified)
            needs_reread = False
            if not is_file_in_chroma(file_path):
                needs_reread = True
            elif file_path in cache and cache[file_path]["last_modified"] != last_modified:
                needs_reread = True
                # Remove old entries from Chroma if modified
                logger.info(f"Deleting old Chroma entries for {filename}")
                chroma_db.delete(where={"filename": filename})

            if file_path in cache and not needs_reread:
                cached_data = cache[file_path]
                text_content = cached_data["text_content"]
                word_count = cached_data["word_count"]
                image_content = cached_data.get("image_content", [])
                file_message = cached_data.get("file_message", "")  # Get reading info from cache

                if do_read_image and image_content == "no image":  # Skip re-extraction if 'no image' is recorded and do_read_image is active
                    pass
                else:
                    # Re-extract if images were not previously stored and do_read_image is True
                    if do_read_image and not image_content:  # and not file_ext=="txt"
                        text_content, image_content, word_count, file_message = extract_content_from_file(file_path,
                                                                                                          do_read_image,
                                                                                                          tesseract_path)

                        # Update the cache with new images or 'no image'
                        cache[file_path]["image_content"] = image_content if image_content else "no image"
                        cache[file_path]["file_message"] = file_message  # Update OCR info
                        save_document_cache(cache)
                        new_files_read += 1  # Increment new file count if reprocessed
            else:
                # Extract text and images, then update the cache
                text_content, image_content, word_count, file_message = extract_content_from_file(file_path,
                                                                                                  do_read_image,
                                                                                                  tesseract_path)

                if do_read_image and not image_content:
                    image_content = "no image"

                cache[file_path] = {
                    "last_modified": last_modified,
                    "text_content": text_content,
                    "word_count": word_count,
                    "image_content": image_content,  # Cache images or 'no image'
                    "file_message": file_message,  # Cache reading info
                }
                save_document_cache(cache)
                new_files_read += 1  # Increment counter for new files

            # Structure the document content as a JSON object for cache (not for Chroma)
            document_json = {
                "filename": filename,
                "pages": [
                    {
                        "page_number": i + 1,
                        "content": page_content["content"] if isinstance(page_content, dict) else page_content,
                        "images": image_content if do_send_images and image_content != "no image" else []
                    }
                    for i, page_content in enumerate(text_content)
                ]
            }

            # Append the JSON object to the document_text list (for cache/UI display)
            document_text.append(document_json)

            cursor = chat_history.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)  # Move cursor to the end
            cursor.insertText(f"✔ Looked at: {filename}\n", system_format)  # Use system_format
            cursor.insertText(f"Word count: {word_count}\n", system_format)  # Use system_format
            if file_message:
                cursor.insertText(f"Alert: {file_message}\n", system_format)  # Add reading info
            if do_read_image:
                cursor.insertText(f"Extracted images: {len(image_content) if image_content != 'no image' else 0}\n", system_format)  # Use system_format
            chat_history.setTextCursor(cursor)  # Update the cursor position
            chat_history.ensureCursorVisible()  # Scroll to the bottom

            # Add images to document_images list if model supports images
            if do_send_images and image_content != "no image":
                document_images.extend(
                    image_content if not isinstance(image_content[0], dict) else [img["content"] for img in
                                                                                  image_content])
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

        # Retrieve relevant document chunks from Chroma
        retriever_docs = chroma_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.3}
        )
        retriever_mem = chroma_db_mem.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 2, "score_threshold": 0.2}
        )

        # Combine the chat history and document context
        messages = [
            {"role": "system", "content": document_text_intro}] if 'document_text_intro' in globals() else []
        messages.append({"role": "user", "content": user_input})

        # If the model supports images, include them in the messages
        if do_send_images:
            for img_base64 in document_images:
                messages.append({"role": "user", "content": f"data:image/png;base64,{img_base64}"})

        try:
            # Start tracking elapsed time
            global start_time
            start_time = time.time()
            # Call update_waiting_label to set the initial text and start the timer
            update_waiting_label()

            # Change the button to "Kill Process"
            send_button.setText("Cancel")
            send_button.setStyleSheet("background-color: red; color: white;")
            send_button.clicked.disconnect()  # Disconnect the previous connection
            send_button.clicked.connect(kill_ollama_process)  # Connect to kill function
            user_input_box.setEnabled(False)

            # Create and start the worker thread with LangChain chain
            global ollama_worker
            ollama_worker = OllamaWorkerThread(
                model_var.currentText(),
                messages,
                {"temperature": temperature_scale.value() / 100.0},
                None,  # Chain built inside thread
                memory,
                retriever_docs,
                retriever_mem,
                document_images,
                do_send_images
            )
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
            model_var.currentText(),
            temperature_scale.value() / 100.0,
            do_mention_page=do_mention_page,
            do_read_image=do_read_image
        )


# Function to handle AI response
def handle_ai_response(response):
    # Stop the metrics timer
    if hasattr(update_waiting_label, "timer_started"):
        metrics_timer.stop()
        delattr(update_waiting_label, "timer_started")  # Reset the timer flag
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
    # Revert the typehere_label to its initial text and style
    typehere_label.setText(INITIAL_TYPEHERE_TEXT)
    typehere_label.setStyleSheet(INITIAL_TYPEHERE_STYLE)


# Function to handle AI errors
def handle_ai_error(error_message):
    # Stop the metrics timer
    if hasattr(update_waiting_label, "timer_started"):
        metrics_timer.stop()
        delattr(update_waiting_label, "timer_started")  # Reset the timer flag

    QMessageBox.critical(window, "Error", f"An error occurred: {error_message}")
    reset_ask_ai_button()

    # Revert the typehere_label to its initial text and style
    typehere_label.setText(INITIAL_TYPEHERE_TEXT)
    typehere_label.setStyleSheet(INITIAL_TYPEHERE_STYLE)


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
    # Revert the typehere_label to its original text and style
    typehere_label.setText("Chat with A.I. here: (Ctrl+↵ to send | Ctrl+^ to revise previous)")
    typehere_label.setStyleSheet("color: green;")


def update_waiting_label():
    # Set the initial "Waiting..." text and style
    typehere_label.setText("Waiting...")
    typehere_label.setStyleSheet("color: orange; font-style: italic;")

    # Start a QTimer to update the label with system metrics (only if not already started)
    global metrics_timer
    if not hasattr(update_waiting_label, "timer_started"):
        metrics_timer = QTimer()
        metrics_timer.timeout.connect(update_waiting_label_metrics)  # Use a separate function for metrics updates
        metrics_timer.start(500)  # Update every 500ms
        update_waiting_label.timer_started = True  # Mark the timer as started


def update_waiting_label_metrics():
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    elapsed_time_str = ""
    if hours > 0:
        elapsed_time_str += f"{hours}h "
    if minutes > 0 or hours > 0:  # Show minutes if hours are present
        elapsed_time_str += f"{minutes}m "
    elapsed_time_str += f"{seconds}s"

    # Get CPU and GPU utilization
    cpu_usage, gpu_usage = get_system_metrics()

    typehere_label.setText(
        f"Waiting... | Elapsed Time: {elapsed_time_str} | CPU: {cpu_usage}% | GPU: {gpu_usage}%"
    )


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 400, 200)  # Set window size

        # Layout
        layout = QVBoxLayout()

        # Label for Tesseract folder
        tesseract_label = QLabel("Tesseract Folder Path:")
        layout.addWidget(tesseract_label)

        # QLineEdit for Tesseract folder path
        self.tesseract_path_edit = QLineEdit()
        self.tesseract_path_edit.setText(tesseract_folder)  # Set current path
        layout.addWidget(self.tesseract_path_edit)

        # Browse button for Tesseract folder
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_tesseract_folder)
        layout.addWidget(browse_button)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def browse_tesseract_folder(self):
        # Open a folder dialog to select the Tesseract folder
        folder = QFileDialog.getExistingDirectory(self, "Select Tesseract Folder")
        if folder:
            self.tesseract_path_edit.setText(folder)
            global tesseract_folder, tesseract_path
            tesseract_folder = folder  # Update the global variable
            tesseract_path = os.path.join(tesseract_folder, "tesseract.exe")  # Update the Tesseract executable path
            print(f"Selected Tesseract folder: {tesseract_folder}")  # Debugging

    def closeEvent(self, event):
        # Save the tesseract_folder to the user data cache when the window is closed
        global tesseract_folder
        user_data = load_user_data()
        user_data["tesseract_folder"] = tesseract_folder
        save_user_data(**user_data)
        print(f"Saving Tesseract folder to cache: {tesseract_folder}")  # Debugging
        event.accept()

    def browse_tesseract_folder(self):
        # Open a folder dialog to select the Tesseract folder
        folder = QFileDialog.getExistingDirectory(self, "Select Tesseract Folder")
        if folder:
            self.tesseract_path_edit.setText(folder)
            global tesseract_folder, tesseract_path
            tesseract_folder = folder  # Update the global variable
            tesseract_path = os.path.join(tesseract_folder, "tesseract.exe")  # Update the Tesseract executable path

    def closeEvent(self, event):
        global tesseract_folder, tesseract_path  # Declare globals explicitly
        user_data = load_user_data()
        user_data["tesseract_folder"] = self.tesseract_path_edit.text()  # Get from the edit box
        save_user_data(**user_data)
        tesseract_folder = self.tesseract_path_edit.text()  # Update the global variable
        tesseract_path = os.path.join(tesseract_folder, "tesseract.exe")
        event.accept()

    def browse_tesseract_folder(self):
        # Open a folder dialog to select the Tesseract folder
        folder = QFileDialog.getExistingDirectory(self, "Select Tesseract Folder")
        if folder:
            self.tesseract_path_edit.setText(folder)
            global tesseract_folder, tesseract_path
            tesseract_folder = folder  # Update the global variable
            tesseract_path = os.path.join(tesseract_folder, "tesseract.exe")  # Update the Tesseract executable path


def open_options():
    settings_dialog = SettingsDialog(window)
    result = settings_dialog.exec()  # Use exec() and store the result

    if result == QDialog.DialogCode.Accepted:  # Check if the user clicked "OK" or closed the dialog with "X"
        global tesseract_folder, tesseract_path  # Declare globals in the function
        user_data = load_user_data()  # Reload user data
        tesseract_folder = user_data.get("tesseract_folder",
                                         r"C:\Program Files\Tesseract-OCR")  # Get from cache or default
        tesseract_path = os.path.join(tesseract_folder, "tesseract.exe")  # Update tesseract_path


def get_system_metrics():
    # Get CPU utilization
    cpu_usage = psutil.cpu_percent(interval=0.1)  # CPU usage in percentage

    # Get GPU utilization using pynvml
    gpu_usage = "N/A"
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use the first GPU
        gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_usage = gpu_info.gpu  # GPU usage in percentage
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass  # GPU not available or error occurred

    return cpu_usage, gpu_usage


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
        # Stop the metrics timer
        if hasattr(update_waiting_label, "timer_started"):
            metrics_timer.stop()
            delattr(update_waiting_label, "timer_started")  # Reset the timer flag


# Function to clear the chat history box
def clear_chat_history():
    chat_history.clear()
    chat_history_list.clear()  # Clear the chat history list
    memory.clear()  # Clear memory as well


# Function to set the folder path
def set_folder_path():
    global document_text
    global document_text_intro
    document_text = []  # Clear the previous document text
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
    document_text_intro, new_files_read, document_images = read_documents(folder_path)
    cursor = chat_history.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)  # Move cursor to the end
    cursor.insertText(f"Documents reading finished. {new_files_read} items were new for the library.\n",
                      other_system_format)
    cursor.insertText("You can now chat with the A.I. \n", other_system_format)
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

    def is_valid_word(self, word):  # Ensure the word contains only English letters (A-Z, a-z)
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
        model_description_label.setText("←Multimodal model detected (you can enable image reading)")
        model_description_label.setStyleSheet("color: orange;")
    else:
        model_description_label.setText("←Standard text-based model")
        model_description_label.setStyleSheet("color: gray;")


def on_toggle(var_name):
    """Update global variables dynamically based on checkbox state."""
    globals()[var_name] = globals()[f"{var_name}_var"].isChecked()  # Access the QCheckBox variable
    # print(f"{var_name}: {globals()[var_name]}")  # Debugging output


# -------------------------------------------------


previous_message = ""
do_revise = False
do_send_images = False  # Initialize the boolean variable
ollama_worker = None
memory = CustomChatMemory()  # Initialize memory
WEB_SEARCH_ENABLED = False  # Toggle for web search

# Define the initial text and style for typehere_label
INITIAL_TYPEHERE_TEXT = "Chat with A.I. here: (Ctrl+↵ to send | Ctrl+^ to revise previous)"
INITIAL_TYPEHERE_STYLE = "color: green;"
# Define the style for the "Waiting..." state
waiting_format = QTextCharFormat()
waiting_format.setForeground(QColor("orange"))  # Orange text color
waiting_format.setFontItalic(True)  # Italic font

other_system_format = QTextCharFormat()
other_system_format.setForeground(QColor("#0000FF"))
other_system_format.setFontWeight(QFont.Weight.Bold)  # Bold text
other_system_format.setFontItalic(True)  # Italic text

# Create the main window
app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("AI Document Analyzer")

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
tesseract_folder = user_data.get("tesseract_folder", r"C:\Program Files\Tesseract-OCR")  # Default path if not in cache
tesseract_path = os.path.join(tesseract_folder, "tesseract.exe")

# Dropdown for model selection
model_var = QComboBox()
model_var.addItems(installed_models)
model_var.setCurrentText(
    last_model if last_model in installed_models else (installed_models[0] if installed_models else "No models found"))

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
# read_button.clicked.connect(lambda: (read_button.setText("Reading..."),set_folder_path()))
top_layout.addWidget(read_button)

# Chat history display
chat_history = QTextEdit()
chat_history.setReadOnly(True)
main_layout.addWidget(chat_history)

# Sample text label
typehere_label = QLabel("Chat with A.I. here: (Ctrl+↵ to send | Ctrl+^ to revise previous)")
typehere_label.setStyleSheet("color: green;")
main_layout.addWidget(typehere_label)

# User input box
user_input_box = SpellCheckTextEdit()
user_input_box.setMaximumHeight(100)
user_input_box.setPlaceholderText("Please read documents first.")  # Set placeholder text
main_layout.addWidget(user_input_box)

# Create the "Options" button
options_button = QPushButton("Options")
options_button.setStyleSheet("background-color: #555555; color: white;")  # Customize appearance
options_button.clicked.connect(open_options)  # Connect to the function

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
    read_button.setStyleSheet("")  # Reset style
    read_button.setText("Read Documents")


def animate_button(button):
    alpha = int((1 + math.sin(time.time() * 2)) * 127.5)  # Sinusoidal alpha
    color = QColor(255, 255, 255, alpha)  # White with varying alpha
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
top_layout.addWidget(model_var)

model_description_label = QLabel("Select a model")
model_description_label.setStyleSheet("color: gray;")
top_layout.addWidget(model_description_label)

# spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
# top_layout.addItem(spacer)
top_layout.addWidget(options_button)

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
disable_ai_interaction()  # for initial app start

custom_font = QFont(QFontDatabase.applicationFontFamilies(font_id)[0]) if (font_id := QFontDatabase.addApplicationFont(
    os.path.join(os.path.dirname(__file__), "modules", "Sahel.ttf"))) != -1 else QFont()
chat_history.setFont(custom_font)
user_input_box.setFont(custom_font)

window.show()
window.setWindowTitle("A.I. Document Analyzer")
# apply style to all disabled buttons in the window:
sys.exit(app.exec())