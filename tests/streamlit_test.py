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
        "chat_history": st.session_state.chat_history,
        "editing_index": st.session_state.editing_index,
        "selected_model": st.session_state.selected_model,
        "new_input_key": st.session_state.new_input_key,
    }
    with open(file_path, "w") as f:
        json.dump(session_data, f)

# Function to load session state from a JSON file
def load_session_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            session_data = json.load(f)
        st.session_state.chat_history = session_data.get("chat_history", [])
        st.session_state.editing_index = session_data.get("editing_index", None)
        st.session_state.selected_model = session_data.get("selected_model", "")
        st.session_state.new_input_key = session_data.get("new_input_key", "new_input_0")
    else:
        st.session_state.chat_history = []
        st.session_state.editing_index = None
        st.session_state.selected_model = ""
        st.session_state.new_input_key = "new_input_0"

# Define the file path for saving/loading session state
SESSION_FILE = "session_state.json"

# Load session state from the file
load_session_state(SESSION_FILE)

# Sidebar for model selection
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

st.title("Chat with Ollama")

# Chat UI
for i in range(len(st.session_state.chat_history)):
    user_text, ai_text = st.session_state.chat_history[i]
    with st.chat_message("user"):
        if st.session_state.editing_index == i:
            new_input = st.text_area("Edit your input:", user_text, key=f"edit_{i}")
            if st.button("Submit Edit", key=f"submit_{i}"):
                st.session_state.chat_history = st.session_state.chat_history[:i]  # Remove subsequent history
                st.session_state.chat_history.append((new_input, ""))  # Update input, reset AI response
                st.session_state.editing_index = None
                # Fetch new AI response
                model = st.session_state.selected_model
                if model:
                    with st.spinner("Generating response..."):
                        response = ollama.chat(model=model, messages=[{"role": "user", "content": new_input}])
                        ai_response = response["message"]["content"] if "message" in response else "Error in response"
                        st.session_state.chat_history[-1] = (new_input, ai_response)
                save_session_state(SESSION_FILE)  # Save session state after updating
                st.rerun()
        else:
            st.markdown(user_text)
            # Add "Edit" and "Delete" buttons
            col1, col2 = st.columns([1, 1])
            if col1.button("Edit", key=f"edit_btn_{i}"):
                st.session_state.editing_index = i
                st.rerun()
            if col2.button("Delete", key=f"delete_btn_{i}"):
                st.session_state.chat_history.pop(i)  # Remove the message and its response
                save_session_state(SESSION_FILE)  # Save session state immediately
                st.rerun()
    with st.chat_message("assistant"):
        st.markdown(ai_text)

# User input
user_input = st.text_area("Enter your message:", key=st.session_state.new_input_key)
if st.button("Send") and user_input.strip():
    model = st.session_state.selected_model
    if model:
        with st.spinner("Generating response..."):
            response = ollama.chat(model=model, messages=[{"role": "user", "content": user_input}])
            ai_response = response["message"]["content"] if "message" in response else "Error in response"
            st.session_state.chat_history.append((user_input, ai_response))
        # Reset the input field by updating the key
        st.session_state.new_input_key = f"new_input_{len(st.session_state.chat_history)}"
        save_session_state(SESSION_FILE)  # Save session state after updating
        st.rerun()
    else:
        st.warning("Please select a model first.")

# Save session state on app exit
st.session_state.save_on_exit = True
if st.session_state.save_on_exit:
    save_session_state(SESSION_FILE)