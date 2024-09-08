import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np


model = YOLO("C:/Users/surya/OneDrive/Desktop/archive(4)/best.pt")

def detect_objects(image):
    image_np = np.array(image)
    results = model.predict(image_np)
    return results

def main():
    st.title("Pothole Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting potholes...")

        
        results = detect_objects(image)


        annotated_image = image.copy()
        for result in results:
            bbox = result['bbox']
            cv2.rectangle(
                np.array(annotated_image),
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0), 2
            )

        st.image(annotated_image, caption='Processed Image.', use_column_width=True)

if __name__ == "__main__":
    main()
