import streamlit as st
import ollama


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


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores (input_text, response_text)
if "editing_index" not in st.session_state:
    st.session_state.editing_index = None  # Index of the input being edited
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""  # Stores selected model

# Sidebar for model selection
with st.sidebar:
    st.markdown("### Settings")
    models = get_installed_models()
    if models:
        st.session_state.selected_model = st.selectbox("Select Model", models, index=0)
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

                st.rerun()
        else:
            st.markdown(user_text)
            if st.button("Edit", key=f"edit_btn_{i}"):
                st.session_state.editing_index = i
                st.rerun()

    with st.chat_message("assistant"):
        st.markdown(ai_text)

# User input
user_input = st.text_area("Enter your message:", key="new_input")
if st.button("Send") and user_input.strip():
    model = st.session_state.selected_model
    if model:
        with st.spinner("Generating response..."):
            response = ollama.chat(model=model, messages=[{"role": "user", "content": user_input}])
            ai_response = response["message"]["content"] if "message" in response else "Error in response"
            st.session_state.chat_history.append((user_input, ai_response))
            st.rerun()
    else:
        st.warning("Please select a model first.")