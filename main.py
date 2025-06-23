import streamlit as st
import os
from PIL import Image, UnidentifiedImageError
from modal_helper import predict

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to disk and return the file path."""
    file_path = "temp_file.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def validate_image(path: str) -> bool:
    """Check if the image at path is valid."""
    try:
        img = Image.open(path)
        img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def main():
    st.set_page_config(page_title="Garbage Classification", layout="centered")
    st.title("🗑️ Garbage Classification")

    st.markdown(
        """
        Upload an image of garbage, and the model will classify it into the correct category.
        Supported formats: **JPG** and **PNG**
        """
    )

    uploaded_file = st.file_uploader("📤 Upload an image file", type=["jpg", "png"])

    if uploaded_file:
        image_path = save_uploaded_file(uploaded_file)

        if validate_image(image_path):
            st.success("✅ Image loaded and verified successfully!")

            # Resize and display image
            img = Image.open(image_path)
            img_resized = img.resize((300, 300))
            st.image(img_resized, caption="Uploaded Image (resized)", use_column_width=False)

            with st.spinner("🔍 Classifying image..."):
                prediction = predict(image_path)

            st.info(f"🧠 **Predicted Class:** `{prediction}`")
        else:
            st.error("⚠️ Invalid image file. Please upload a valid JPG or PNG image.")

if __name__ == "__main__":
    main()
