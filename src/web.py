import streamlit as st
from query import ask  # Adjust the import path if necessary

# Streamlit App
st.title("AdiletLLM Question Answering")

# Input Section
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        # Show a spinner while the `ask` function is processing
        with st.spinner("Thinking..."):
            try:
                # Call the `ask` function and display the result
                answer = ask(question)
                st.success(f"Answer: {answer}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")
