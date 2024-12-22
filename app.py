import streamlit as st
import torch
import pickle
from vqa_model import VQAModel
import os
from PIL import Image
import numpy as np

# Loading the fitted One Hot Encoders from the disk
with open('Saved_Models/answer_onehotencoder.pkl', 'rb') as f:
    ANSWER_ONEHOTENCODER = pickle.load(f)
with open('Saved_Models/answer_type_onehotencoder.pkl', 'rb') as f:
    ANSWER_TYPE_ONEHOTENCODER = pickle.load(f)

# Loading the model from the disk
DEVICE = torch.device("cpu")
MODEL_NAME = "ViT-L/14@336px"
NUM_CLASSES = 5410
MODEL_PATH = "Saved_Models/epoch_50.pth"
model = VQAModel(num_classes=NUM_CLASSES, device=DEVICE, hidden_size=512, model_name=MODEL_NAME).to(DEVICE)
model.load_model(MODEL_PATH)

# Streamlit UI
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
            # Save the uploaded file temporarily
            image_path = "static/uploads/user_image.jpg"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            # Open the image and save it
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Prediction
            predicted_answer, predicted_answer_type, answerability = model.test_model(image_path=image_path, question=question)
            answer = ANSWER_ONEHOTENCODER.inverse_transform(predicted_answer.cpu().detach().numpy())
            answer_type = ANSWER_TYPE_ONEHOTENCODER.inverse_transform(predicted_answer_type.cpu().detach().numpy())

            # Display results
            st.image(Image.open(image_path), caption="Uploaded Image", use_column_width=True)
            st.write("### Predicted Answer")
            st.write(f"**Answer:** {answer[0][0]}")
            st.write(f"**Answer Type:** {answer_type[0][0]}")
            st.write(f"**Answerability:** {answerability.item():.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
