import streamlit as st
import requests
from PIL import Image
import io

# Set the API endpoint
API_ENDPOINT = "https://your-model-api-endpoint/predict"

st.title("Visual Question Answering (VQA) Webapp")
st.write("Upload an image and ask a question about it.")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not uploaded_file:
        st.error("Please upload an image.")
    elif not question:
        st.error("Please enter a question.")
    else:
        try:
            # Send image and question to the API
            files = {"file": uploaded_file.getvalue()}
            data = {"question": question}
            response = requests.post(API_ENDPOINT, files=files, data=data)
            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                # Display the image
                image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Display results
                st.write("### Predicted Answer")
                st.write(f"**Answer:** {result['answer']}")
                st.write(f"**Answer Type:** {result['answer_type']}")
                st.write(f"**Answerability:** {result['answerability']:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
