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

def generate_response(model, prompt):
    """Get a response from the selected Ollama model."""
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""


def main():
    st.set_page_config(page_title="Ollama Chat", layout="wide")

    st.sidebar.header("Settings")
    models = get_installed_models()
    if models:
        selected_model = st.sidebar.selectbox("Select a model", models, index=0)
    else:
        st.sidebar.warning("No models found. Ensure Ollama is running and models are installed.")
        selected_model = None

    st.title("Ollama Chat Interface")

    if selected_model:
        user_input = st.text_area("Enter your message:", height=150)
        if st.button("Send") and user_input.strip():
            with st.spinner("Generating response..."):
                response = generate_response(selected_model, user_input)
                st.text_area("Response:", value=response, height=200, disabled=True)
    else:
        st.warning("Please select a model to continue.")


if __name__ == "__main__":
    main()