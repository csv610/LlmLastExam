import os
import streamlit as st
from datasets import load_dataset
import google.generativeai as genai

def configure_genai():
    """Configure and return the Gemini model."""
    google_api_key = os.getenv('GEMINI_API_KEY')
    if not google_api_key:
        st.error("Please set the GEMINI_API_KEY environment variable")
        st.stop()
    
    genai.configure(api_key=google_api_key)
    return genai.GenerativeModel("gemini-2.0-flash-exp")

def load_data():
    """Load the HLE dataset."""
    try:
        return load_dataset("cais/hle", split='test')
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

def get_selected_question(dataset):
    """Handle question selection and navigation."""
    if 'index' not in st.session_state:
        st.session_state.index = 0
    
    st.sidebar.title("Exam Info")
    st.sidebar.write(f"Total Questions: {len(dataset)}")
    
    # First place the slider
    selected_index = st.sidebar.slider(
        "Select Question ID", 
        0, 
        len(dataset) - 1, 
        st.session_state.index
    )
    st.session_state.index = selected_index
    
    # Add a small space
    st.sidebar.write("")
    
    # Then place the navigation buttons
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Previous"):
        st.session_state.index = max(0, st.session_state.index - 1)
        st.rerun()
    if col2.button("Next"):
        st.session_state.index = min(len(dataset) - 1, st.session_state.index + 1)
        st.rerun()
    
    return dataset[st.session_state.index]

def display_question(row, model):
    """Display the question, handle model interaction, and show results."""
    question = row.get('question', None)
    image = row.get('image', None)
    
    # Display question
    st.subheader(f"Question {st.session_state.index}:")
    st.write(question)
    
    # Display image if available
    if image:
        st.image(image, caption="Input Image", use_container_width=True)
    
    # Model response section
    if st.sidebar.button("Get Response"):
        with st.spinner("Generating response..."):
            try:
                if image:
                    response = model.generate_content([question, image])
                else:
                    response = model.generate_content(question)
                st.subheader("Model Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Answer reveal section
    if st.sidebar.button("Get Correct Answer"):
        answer = row.get('answer', None)
        if answer:
            st.subheader("Correct Answer:")
            st.write(answer)
        else:
            st.warning("No answer available for this question")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Humanity Last Exam",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("Humanity Last Exam")
    
    try:
        model = configure_genai()
        dataset = load_data()
        row = get_selected_question(dataset)
        display_question(row, model)
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
