import streamlit as st
import ollama
import json
import os

# Function to get installed models
def get_installed_models():
    try:
        models_response = ollama.list()  # Fetch the list of installed models
        models_list = getattr(models_response, "models", [])
        if not isinstance(models_list, list):
            raise ValueError("Unexpected response format: 'models' attribute is not a list")
        model_names = [getattr(model, "model", "Unknown") for model in models_list]
        return model_names
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

# Function to save session state to a JSON file
def save_session_state(file_path):
    session_data = {
        "chats": st.session_state.chats,
        "current_chat_index": st.session_state.current_chat_index,
        "selected_model": st.session_state.selected_model,
    }
    with open(file_path, "w") as f:
        json.dump(session_data, f)

# Function to load session state from a JSON file
def load_session_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            session_data = json.load(f)
        st.session_state.chats = session_data.get("chats", [{"name": "Chat 1", "history": []}])
        st.session_state.current_chat_index = session_data.get("current_chat_index", 0)
        st.session_state.selected_model = session_data.get("selected_model", "")
    else:
        st.session_state.chats = [{"name": "Chat 1", "history": []}]
        st.session_state.current_chat_index = 0
        st.session_state.selected_model = ""

# Define the file path for saving/loading session state
SESSION_FILE = "session_state.json"

# Load session state from the file
load_session_state(SESSION_FILE)

# Sidebar for model selection and chat management
with st.sidebar:
    st.markdown("### Settings")
    models = get_installed_models()
    if models:
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            models,
            index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
        )
    else:
        st.warning("No models found. Please install a model.")

    # Ensure at least one chat exists
    if not hasattr(st.session_state, "chats") or not st.session_state.chats:
        st.session_state.chats = [{"name": "Chat 1", "history": []}]
        st.session_state.current_chat_index = 0

    st.markdown("### Chats")

    for index, chat in enumerate(st.session_state.chats):
        col1, col2 = st.columns([4, 1])  # Chat name and options button
        if st.session_state.current_chat_index == index:
            col1.markdown(f"**{chat['name']}**")  # Highlight active chat
        else:
            col1.write(chat["name"])  # Normal display

        with col2:
            if st.button("⋮", key=f"menu_{index}"):
                st.session_state[f"menu_open_{index}"] = not st.session_state.get(f"menu_open_{index}", False)

        if st.session_state.get(f"menu_open_{index}", False):
            with st.expander("Chat Options", expanded=True):
                # Rename chat
                new_name = st.text_input("Rename chat:", value=chat["name"], key=f"rename_{index}")
                if st.button("Save", key=f"save_rename_{index}"):
                    st.session_state.chats[index]["name"] = new_name
                    st.session_state[f"menu_open_{index}"] = False
                    save_session_state(SESSION_FILE)
                    st.rerun()

                # Delete chat (Allows deleting last one, but creates a new default if empty)
                if st.button("Delete Chat", key=f"delete_{index}"):
                    del st.session_state.chats[index]

                    if not st.session_state.chats:  # If no chats left, create a default one
                        st.session_state.chats = [{"name": "Chat 1", "history": []}]
                        st.session_state.current_chat_index = 0
                    else:
                        st.session_state.current_chat_index = min(index, len(st.session_state.chats) - 1)

                    st.session_state[f"menu_open_{index}"] = False
                    save_session_state(SESSION_FILE)
                    st.rerun()

    # Add new chat button at the bottom
    if st.button("New Chat"):
        new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats.append({"name": new_chat_name, "history": []})
        st.session_state.current_chat_index = len(st.session_state.chats) - 1
        save_session_state(SESSION_FILE)
        st.rerun()

st.title("Chat with Ollama")

# Get the current chat
current_chat = st.session_state.chats[st.session_state.current_chat_index]

# Chat UI
for i in range(len(current_chat["history"])):
    user_text, ai_text = current_chat["history"][i]
    with st.chat_message("user"):
        st.markdown(user_text)
        col1, col2 = st.columns([1, 1])
        if col1.button("Edit", key=f"edit_btn_{i}"):
            st.session_state.editing_index = i
            st.rerun()
        if col2.button("Delete", key=f"delete_btn_{i}"):
            current_chat["history"].pop(i)
            save_session_state(SESSION_FILE)
            st.rerun()
    with st.chat_message("assistant"):
        st.markdown(ai_text)

# User input
if "editing_index" not in st.session_state:
    st.session_state.editing_index = None

if st.session_state.editing_index is not None:
    edit_index = st.session_state.editing_index
    user_text, _ = current_chat["history"][edit_index]
    new_input = st.text_area("Edit your input:", user_text, key=f"edit_input_{st.session_state.current_chat_index}")
    if st.button("Submit Edit", key=f"submit_edit_{st.session_state.current_chat_index}"):
        current_chat["history"] = current_chat["history"][:edit_index]  # Remove subsequent history
        current_chat["history"].append((new_input, ""))  # Update input, reset AI response
        st.session_state.editing_index = None
        model = st.session_state.selected_model
        if model:
            with st.spinner("Generating response..."):
                response = ollama.chat(model=model, messages=[{"role": "user", "content": new_input}])
                ai_response = response["message"]["content"] if "message" in response else "Error in response"
                current_chat["history"][-1] = (new_input, ai_response)
        save_session_state(SESSION_FILE)
        st.rerun()
else:
    # Use a unique key for the input text area based on the current chat index
    user_input = st.text_area("Enter your message:", key=f"new_input_{st.session_state.current_chat_index}")
    if st.button("Send", key=f"send_{st.session_state.current_chat_index}") and user_input.strip():
        model = st.session_state.selected_model
        if model:
            with st.spinner("Generating response..."):
                response = ollama.chat(model=model, messages=[{"role": "user", "content": user_input}])
                ai_response = response["message"]["content"] if "message" in response else "Error in response"
                current_chat["history"].append((user_input, ai_response))
            save_session_state(SESSION_FILE)
            st.rerun()
        else:
            st.warning("Please select a model first.")

# Save session state on app exit
st.session_state.save_on_exit = True
if st.session_state.save_on_exit:
    save_session_state(SESSION_FILE)