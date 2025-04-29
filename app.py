import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.5rem;
        font-weight: 500;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .image-container {
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(237, 242, 247, 0.5);
    }
    .app-footer {
        text-align: center;
        margin-top: 2rem;
        opacity: 0.7;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_age_gender_model():
    try:
        model = load_model("best_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(uploaded_image):
    if uploaded_image.mode != "L":
        uploaded_image = uploaded_image.convert("L")
    image = uploaded_image.resize((128, 128))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    return np.expand_dims(image_array, axis=0)

def predict_age_gender(model, image_array):
    try:
        predictions = model.predict(image_array)
        predicted_age = int(np.round(predictions[1][0]))
        gender_prob = predictions[0][0]
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        gender_confidence = gender_prob if predicted_gender == "Female" else 1 - gender_prob
        return predicted_age, predicted_gender, float(gender_confidence)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def live_camera_detection(model):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    exit_button_pressed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (128, 128))
            face_array = np.expand_dims(np.expand_dims(face_resized, axis=-1), axis=0) / 255.0
            age, gender, confidence = predict_age_gender(model, face_array)
            if age is not None and gender is not None:
                label = f"{gender}, {age} yrs, {confidence:.0%}"
                color = (255, 0, 255) if gender == "Female" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        stframe.image(frame, channels="BGR")

        if st.button("Exit Live Camera", key="exit_live_camera_button"):
            exit_button_pressed = True
            break

    cap.release()
    cv2.destroyAllWindows()
    if exit_button_pressed:
        st.success("Live camera stopped.")

def main():
    st.markdown('<div class="main-header">Age and Gender Detector</div>', unsafe_allow_html=True)
    with st.spinner("Loading model... This may take a moment."):
        model = load_age_gender_model()

    if model is None:
        st.warning("Please make sure the model file exists at the specified path.")
        return

    st.markdown('<div class="sub-header">Upload Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files and st.button("Detect Age & Gender", key="upload_button"):
        with st.spinner("Analyzing images..."):
            for i, uploaded_file in enumerate(uploaded_files):
                with st.container():
                    st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                    st.markdown(f"<h3>Image {i+1}</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 1])

                    image = Image.open(uploaded_file)
                    col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_column_width=True)
                    processed_image = preprocess_image(image)
                    age, gender, confidence = predict_age_gender(model, processed_image)

                    if age is not None and gender is not None:
                        col2.markdown('<div class="sub-header">Results:</div>', unsafe_allow_html=True)
                        col2.markdown(f'<div class="result-text" style="background-color: rgba(37, 99, 235, 0.1);">Age: {age}</div>', unsafe_allow_html=True)
                        gender_color = "#9F7AEA" if gender == "Female" else "#4F46E5"
                        col2.markdown(f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(gender_color)))}, 0.1);">Gender: {gender}<br><small>Confidence: {confidence:.2%}</small></div>', unsafe_allow_html=True)
                    else:
                        col2.error("Failed to process this image")

                    st.markdown("</div>", unsafe_allow_html=True)
                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

    elif st.button("Detect Age & Gender", key="detect_button"):
        st.info("Please upload one or more images first.")

    st.markdown('<div class="sub-header">Take a Snapshot from Camera</div>', unsafe_allow_html=True)

    if 'show_camera' not in st.session_state:
        st.session_state.show_camera = False

    if st.button("Open Snapshot Camera", key="open_camera_button"):
        st.session_state.show_camera = True

    if st.session_state.show_camera:
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            if st.button("Detect from Snapshot", key="detect_from_snapshot_button"):
                image = Image.open(camera_image)
                processed_image = preprocess_image(image)
                age, gender, confidence = predict_age_gender(model, processed_image)
                if age is not None and gender is not None:
                    st.image(image, caption="Snapshot", use_column_width=True)
                    st.markdown(f"**Age**: {age}")
                    st.markdown(f"**Gender**: {gender} ({confidence:.2%} confidence)")
        if st.button("Exit Snapshot", key="exit_snapshot_button"):
            st.session_state.show_camera = False
            st.rerun()

    st.markdown('<div class="sub-header">Live Camera Detection</div>', unsafe_allow_html=True)
    if st.button("Start Live Camera", key="start_live_camera_button"):
        st.info("Click 'Exit Live Camera' to stop.")
        live_camera_detection(model)

    if st.button("Exit App", key="exit_app_button"):
        st.success("Exiting application...")
        st.stop()

    st.markdown('<div class="app-footer">Powered by KANISHK KARAM 💡</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
