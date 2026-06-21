import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Counter App", layout="centered")

st.title("Face Counter App")
st.write("Upload ảnh để hệ thống detect và đếm số khuôn mặt.")

uploaded_file = st.file_uploader(
    "Chọn ảnh",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Ảnh gốc")
    st.image(image_np, use_container_width=True)

    with st.spinner("Đang detect khuôn mặt..."):
        face_locations = face_recognition.face_locations(
            image_np,
            number_of_times_to_upsample=1,
            model="hog"
        )

    result_image = image_np.copy()

    for i, (top, right, bottom, left) in enumerate(face_locations, start=1):
        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(
            result_image,
            f"Face {i}",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    st.subheader("Kết quả")
    st.success(f"Detect được {len(face_locations)} khuôn mặt.")
    st.image(result_image, use_container_width=True)