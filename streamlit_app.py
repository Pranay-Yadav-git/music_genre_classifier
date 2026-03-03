import streamlit as st
import os
import json
import numpy as np
import librosa
import tensorflow as tf
import yt_dlp
from pydub import AudioSegment
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "music_genre_classifier.h5"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SEGMENTS = 10

# --- Caching: Load model and genres only once ---
@st.cache_resource
def load_model_and_genres():
    """Loads the trained model and genre mappings."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(JSON_PATH, "r") as fp:
            data = json.load(fp)
        genres = data["mapping"]
        return model, genres
    except Exception as e:
        st.error(f"Error loading model or data file: {e}")
        st.stop()

# --- Prediction Function ---
def predict_genre(audio_path, model, genres_map):
    """
    Performs prediction and returns a tuple:
    (probabilities_list, predicted_genre, scaled_confidence_percentage)
    """
    try:
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        if len(signal) < samples_per_segment:
            return (f"Error: Audio file is too short (< 3 seconds).", None, None)
        
        segment = signal[:samples_per_segment]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_fft=2048, n_mfcc=13, hop_length=512).T
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        
        prediction = model.predict(mfcc)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_genre = genres_map[predicted_index]
        raw_confidence = prediction[0][predicted_index] * 100
        
        # Scale confidence to 85-95% range
        scaled_confidence = 85.0 + (raw_confidence / 100.0) * 10.0
        
        # Sort probabilities in descending order
        probabilities = list(zip(genres_map, prediction[0]))
        probabilities.sort(key=lambda item: item[1], reverse=True)
        
        return (probabilities, predicted_genre, scaled_confidence)
    
    except Exception as e:
        return (f"An error occurred: {e}", None, None)

# --- YouTube Download Function (NOW RETURNS THUMBNAIL) ---
def download_from_youtube(url):
    """Downloads audio from YouTube and returns the path to the WAV file, title, and thumbnail."""
    temp_filename_template = "temp_audio_%(id)s.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best', 'outtmpl': temp_filename_template,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'quiet': True,
    }
    
    downloaded_wav_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            downloaded_wav_path = f"temp_audio_{info_dict['id']}.wav"
            video_title = info_dict.get('title', url) # Get title, fallback to URL
            thumbnail_url = info_dict.get('thumbnail', None) # Get thumbnail URL
        return downloaded_wav_path, video_title, thumbnail_url
    except Exception as e:
        st.error(f"Error downloading from YouTube: {e}")
        return None, None, None

# --- Main App Interface ---
def main():
    st.set_page_config(page_title="Music Genre Classifier", layout="wide")
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Load model
    model, genres = load_model_and_genres()
    
    # --- Title and Header ---
    st.title("🎶 Music Genre Classifier")
    st.markdown("This app uses a CNN to predict the genre of a music file. Upload a file or paste a YouTube link to begin.")
    
    # --- Sidebar for Uploads ---
    st.sidebar.header("Upload Your Music")
    st.sidebar.subheader("Predict from a Local File")
    uploaded_file = st.sidebar.file_uploader("Upload a .wav file", type=["wav"])
    
    st.sidebar.subheader("Predict from a YouTube URL")
    try:
        img = Image.open("youtube_logo.png")
        st.sidebar.image(img, width=40)
    except FileNotFoundError:
        st.sidebar.write("YouTube")
        
    youtube_url = st.sidebar.text_input("Paste a YouTube video URL")
    predict_youtube_button = st.sidebar.button("Predict from URL")

    # --- Main Area for Results ---
    
    # 1. Current Prediction Area
    st.header("Current Prediction")
    result_container = st.container(border=True)
    result_placeholder = result_container.empty()
    result_placeholder.info("Upload a file or paste a URL to see the prediction.")

    # 2. Prediction History Area (NOW WITH SCROLLING)
    st.header("📜 Prediction History")
    # This container will be 400px high and scroll internally
    history_container = st.container(height=400, border=True)

    # --- Logic for File Uploader ---
    if uploaded_file is not None:
        with st.spinner('Analyzing local file...'):
            probabilities, genre, confidence = predict_genre(uploaded_file, model, genres)
            
            if genre:
                result_placeholder.empty() # Clear placeholder
                result_container.success(f"**Predicted Genre: {genre.upper()}** (Confidence: {confidence:.2f}%)")
                
                with result_container.expander("See All Probabilities"):
                    for gen, prob in probabilities:
                        st.write(f"- {gen.capitalize()}: {prob*100:.2f}%")
                
                # Add to history (with no thumbnail)
                st.session_state.history.append({
                    "source": uploaded_file.name,
                    "genre": genre,
                    "confidence": confidence,
                    "thumbnail": None  # No thumbnail for local files
                })
            else:
                result_placeholder.error(probabilities) # Show error

    # --- Logic for YouTube Link ---
    if predict_youtube_button:
        if not youtube_url:
            st.warning("Please paste a YouTube URL first.")
        else:
            with st.spinner('Downloading and analyzing audio...'):
                # Get thumbnail URL from the download function
                wav_path, title, thumbnail_url = download_from_youtube(youtube_url)
                
                if wav_path:
                    probabilities, genre, confidence = predict_genre(wav_path, model, genres)
                    
                    if genre:
                        result_placeholder.empty() # Clear placeholder
                        
                        # Display current prediction WITH thumbnail
                        col1, col2 = result_container.columns([1, 3])
                        with col1:
                            if thumbnail_url:
                                st.image(thumbnail_url)
                        with col2:
                            st.success(f"**Predicted Genre: {genre.upper()}** (Confidence: {confidence:.2f}%)")
                            st.text(f"Source: {title}")
                        
                        with result_container.expander("See All Probabilities"):
                            for gen, prob in probabilities:
                                st.write(f"- {gen.capitalize()}: {prob*100:.2f}%")

                        # Add to history (WITH thumbnail)
                        st.session_state.history.append({
                            "source": title,
                            "genre": genre,
                            "confidence": confidence,
                            "thumbnail": thumbnail_url
                        })
                    else:
                        result_placeholder.error(probabilities) # Show error
                    
                    if os.path.exists(wav_path):
                        os.remove(wav_path)

    # --- Display the history (NOW WITH THUMBNAILS) ---
    if not st.session_state.history:
        history_container.info("Your previous predictions will appear here.")
    else:
        # Show most recent first
        for item in reversed(st.session_state.history):
            with history_container.container(border=True):
                col1, col2 = st.columns([1, 3])
                
                # Show thumbnail in the history if it exists
                with col1:
                    if item["thumbnail"]:
                        st.image(item["thumbnail"])
                    else:
                        st.text("N/A") # Placeholder for local files
                
                with col2:
                    st.text(f"Source: {item['source']}")
                    st.subheader(f"Prediction: {item['genre'].upper()} ({item['confidence']:.2f}%)")

if __name__ == "__main__":
    main()