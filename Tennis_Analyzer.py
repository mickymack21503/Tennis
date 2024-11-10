import os
import gdown
import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# Function to download model file with validation
def download_file(url, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Download using gdown
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=True)  # Silent download
    
    # Verify the downloaded file is a PyTorch model
    if os.path.getsize(file_path) < 1024 * 1024:  # Check for small file sizes
        os.remove(file_path)
        gdown.download(url, file_path, quiet=True)

# Load YOLO model
@st.cache_resource
def load_model():
    model_path = "models/yolov8x.pt"
    model_url = "https://drive.google.com/uc?export=download&id=1F3k5nOXN-BXs_egDTOd7UOQWpI-9H4Sn"
    
    # Download if not already present
    download_file(model_url, model_path)

    # Load YOLO model
    model = YOLO(model_path)  # Load the YOLO model directly
    return model

# Process video function with preview
def process_video(input_path, output_path, preview=False):
    model = load_model()
    video = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.sidebar.progress(0)

    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame)
        annotated_frame = results[0].plot()  # Get the annotated frame

        # Write annotated frame to output
        out.write(annotated_frame)

        # If preview is enabled, show the frame
        if preview:
            st.image(annotated_frame, channels="BGR", use_column_width=True)

        progress_bar.progress((i + 1) / frame_count)

    video.release()
    out.release()
    progress_bar.empty()
    return os.path.exists(output_path)

# Streamlit UI code
st.title("🎾 Tennis Game Tracking 🎾")
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("📂 Select Input Video File", type=["mp4", "avi", "mov"])

# Option to toggle video preview
preview_option = st.sidebar.checkbox("Show preview during processing", value=False)

if uploaded_file:
    # Save the uploaded file temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(uploaded_file.read())
    temp_input_path = temp_input.name
    temp_output_path = tempfile.mktemp(suffix=".mp4")

    if st.sidebar.button("Process Video"):
        st.sidebar.text("Processing Video...")
        success = process_video(temp_input_path, temp_output_path, preview=preview_option)
        
        if success:
            st.sidebar.text("Processing complete!")
            st.video(temp_output_path)
            with open(temp_output_path, "rb") as file:
                st.sidebar.download_button(
                    label="⬇️ Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.sidebar.error("Processing failed. Could not generate output video.")

    # Cleanup input file
    temp_input.close()
    os.remove(temp_input_path)
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
else:
    st.info("Please upload a video file to start processing.")
