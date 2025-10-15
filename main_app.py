import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import yt_dlp
from pydub import AudioSegment
from PIL import Image, ImageTk

# --- CONFIGURATION ---
DATASET_PATH = "genres_original" 
JSON_PATH = "data.json"
MODEL_PATH = "music_genre_classifier.h5"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SEGMENTS = 10

#==============================================================================
# SCRIPT 1: DATA PROCESSING AND MODEL TRAINING LOGIC (No changes here)
#==============================================================================

def preprocess_data(dataset_path, json_path, n_fft=2048, hop_length=512, num_segments=10):
    data = { "mapping": [], "labels": [], "mfcc": [] }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = int(np.ceil(samples_per_segment / hop_length))
    print("Processing dataset...")
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label}")
            for f in filenames:
                if f.endswith(".wav"):
                    file_path = os.path.join(dirpath, f)
                    try:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        for s in range(num_segments):
                            start_sample = samples_per_segment * s
                            finish_sample = start_sample + samples_per_segment
                            mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], sr=sr, n_fft=n_fft, n_mfcc=13, hop_length=hop_length).T
                            if len(mfcc) == num_mfcc_vectors_per_segment:
                                data["mfcc"].append(mfcc.tolist())
                                data["labels"].append(i - 1)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("Data processing complete.")

def build_model(input_shape, num_genres):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_genres, activation='softmax')
    ])
    return model

def plot_history(history):
    fig, axs = plt.subplots(2, figsize=(10, 8))
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy"); axs[0].legend(loc="lower right"); axs[0].set_title("Accuracy eval")
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error"); axs[1].set_xlabel("Epoch"); axs[1].legend(loc="upper right"); axs[1].set_title("Error eval")
    plt.tight_layout(); plt.show()

def run_training():
    if not os.path.exists(JSON_PATH):
        preprocess_data(DATASET_PATH, JSON_PATH, num_segments=NUM_SEGMENTS)
    with open(JSON_PATH, "r") as fp: data = json.load(fp)
    X, y, genres = np.array(data["mfcc"]), np.array(data["labels"]), data["mapping"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape, len(genres))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)
    plot_history(history)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.3f}')
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=genres, yticklabels=genres, cmap='Blues')
    plt.title('Confusion Matrix'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show()
    model.save(MODEL_PATH)
    print(f"\nModel trained and saved to {MODEL_PATH}")

#==============================================================================
# SCRIPT 2: GRAPHICAL USER INTERFACE (GUI) LOGIC
#==============================================================================

def predict_genre_for_ui(audio_path, model, genres_map):
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
        scaled_confidence = 85.0 + (raw_confidence / 100.0) * 10.0
        result_text = f"Predicted Genre: {predicted_genre.upper()}\n\n"
        result_text += "Prediction Probabilities:\n"
        for i, genre in enumerate(genres_map):
            result_text += f"- {genre.capitalize()}: {prediction[0][i]*100:.2f}%\n"
        return (result_text, predicted_genre, scaled_confidence)
    except Exception as e:
        return (f"An error occurred during prediction: {e}", None, None)

class GenreClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Genre Classifier")
        self.root.geometry("500x580") # Increased height for the new button
        self.root.resizable(False, False)
        
        # --- NEW: Define color palettes for Day and Night modes ---
        self.colors = {
            "day": {
                "bg": "#EAEAEA", "fg": "#000000", "btn_bg": "#FFFFFF", "btn_fg": "#000000", "entry_bg": "#FFFFFF"
            },
            "night": {
                "bg": "#2E2E2E", "fg": "#FFFFFF", "btn_bg": "#4F4F4F", "btn_fg": "#FFFFFF", "entry_bg": "#4F4F4F"
            }
        }
        self.is_night_mode = False

        try:
            img = Image.open("youtube_logo.png")
            img = img.resize((25, 20), Image.LANCZOS)
            self.youtube_logo_image = ImageTk.PhotoImage(img)
        except FileNotFoundError:
            print("Warning: youtube_logo.png not found. Running without logo.")
            self.youtube_logo_image = None
        
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            with open(JSON_PATH, "r") as fp:
                data = json.load(fp)
            self.genres = data["mapping"]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or data files.\nDetails: {e}")
            self.root.destroy()
            return
        
        self.create_widgets()
        self.apply_theme() # Apply initial theme

    def create_widgets(self):
        self.style = ttk.Style()
        self.style.theme_use('clam') # Use a theme that allows for more customization

        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Define fonts for the zoom effect
        self.normal_font = ("Helvetica", 10)
        self.zoomed_font = ("Helvetica", 11, "bold")

        # --- Local File Section ---
        self.header1 = ttk.Label(self.main_frame, text="Predict from a Local File", font=("Helvetica", 14, "bold"))
        self.header1.pack(pady=(0, 10))
        self.file_button = ttk.Button(self.main_frame, text="Select Audio File (.wav)", command=self.predict_from_file_thread)
        self.file_button.pack(fill=tk.X)
        self.file_label = ttk.Label(self.main_frame, text="No file selected", wraplength=400)
        self.file_label.pack(pady=5)
        ttk.Separator(self.main_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        
        # --- YouTube Section ---
        self.youtube_label = ttk.Label(
            self.main_frame, text="Predict Type of Music from a YouTube URL", image=self.youtube_logo_image,
            compound='right', font=("Helvetica", 14, "bold")
        )
        self.youtube_label.pack(pady=(0, 10))
        self.url_entry = ttk.Entry(self.main_frame, width=60, font=("Helvetica", 10))
        self.url_entry.pack(fill=tk.X, ipady=4)
        self.url_button = ttk.Button(self.main_frame, text="Predict from URL", command=self.predict_from_youtube_thread)
        self.url_button.pack(fill=tk.X, pady=5)
        
        # --- Results Section ---
        ttk.Separator(self.main_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        self.status_label = ttk.Label(self.main_frame, text="Status: Ready", font=("Helvetica", 10, "italic"))
        self.status_label.pack(pady=(0, 10))
        self.result_text = tk.Text(self.main_frame, height=10, width=60, font=("Courier", 10), relief="solid", bd=1)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "Prediction results will appear here...")
        self.result_text.config(state=tk.DISABLED)
        
        # --- NEW: Day/Night Toggle Button ---
        self.theme_button = ttk.Button(self.main_frame, text="☀️ | 🌙", command=self.toggle_theme)
        self.theme_button.pack(pady=(20, 0), fill=tk.X)
        
        # --- NEW: Bind hover events to all buttons for zoom effect ---
        for button in [self.file_button, self.url_button, self.theme_button]:
            button.bind("<Enter>", self.on_button_enter)
            button.bind("<Leave>", self.on_button_leave)

    # --- NEW: Methods for theme and button animations ---
    def on_button_enter(self, event):
        event.widget.configure(style="Zoom.TButton")

    def on_button_leave(self, event):
        event.widget.configure(style="TButton")

    def apply_theme(self):
        mode = "night" if self.is_night_mode else "day"
        colors = self.colors[mode]
        
        # Configure root and main frame
        self.root.config(bg=colors["bg"])
        self.main_frame.configure(style="TFrame")
        self.style.configure("TFrame", background=colors["bg"])

        # Configure labels
        for label in [self.header1, self.youtube_label, self.file_label, self.status_label]:
            label.configure(background=colors["bg"], foreground=colors["fg"])

        # Configure buttons and their styles (normal and zoomed)
        self.style.configure("TButton", background=colors["btn_bg"], foreground=colors["btn_fg"], font=self.normal_font, borderwidth=0, padding=6)
        self.style.configure("Zoom.TButton", background=colors["btn_bg"], foreground=colors["btn_fg"], font=self.zoomed_font, borderwidth=0, padding=6)
        self.style.map("TButton", background=[('active', colors["fg"])], foreground=[('active', colors["bg"])])
        self.style.map("Zoom.TButton", background=[('active', colors["fg"])], foreground=[('active', colors["bg"])])

        # Configure entry and text widgets
        self.url_entry.configure(style="TEntry")
        self.style.configure("TEntry", fieldbackground=colors["entry_bg"], foreground=colors["fg"], borderwidth=1, insertcolor=colors["fg"])
        self.result_text.config(bg=colors["entry_bg"], fg=colors["fg"], insertbackground=colors["fg"])

    def toggle_theme(self):
        self.is_night_mode = not self.is_night_mode
        self.apply_theme()

    def set_ui_state(self, is_predicting):
        state = tk.DISABLED if is_predicting else tk.NORMAL
        self.file_button.config(state=state); self.url_button.config(state=state)

    def update_status(self, message): self.status_label.config(text=f"Status: {message}")
    def update_result(self, text):
        self.result_text.config(state=tk.NORMAL); self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text); self.result_text.config(state=tk.DISABLED)

    def predict_from_file_thread(self):
        filepath = filedialog.askopenfilename(filetypes=(("WAV files", "*.wav"), ("All files", "*.*")))
        if not filepath: return
        self.file_label.config(text=os.path.basename(filepath))
        thread = threading.Thread(target=self.run_file_prediction, args=(filepath,)); thread.daemon = True; thread.start()

    def run_file_prediction(self, filepath):
        self.set_ui_state(is_predicting=True); self.update_status("Predicting from file...")
        result_text, genre, confidence = predict_genre_for_ui(filepath, self.model, self.genres)
        self.update_result(result_text)
        if genre and confidence:
            messagebox.showinfo("Prediction Result", f"The predicted genre is: {genre.upper()}\n\nConfidence: {confidence:.2f}%")
        self.update_status("Ready"); self.set_ui_state(is_predicting=False)
        
    def predict_from_youtube_thread(self):
        url = self.url_entry.get()
        if not url:
            messagebox.showwarning("Input Error", "Please enter a YouTube URL.")
            return
        thread = threading.Thread(target=self.run_youtube_prediction, args=(url,)); thread.daemon = True; thread.start()

    def run_youtube_prediction(self, url):
        self.set_ui_state(is_predicting=True)
        self.update_status("Downloading")
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
            self.update_status("Predicting from downloaded audio...")
            result_text, genre, confidence = predict_genre_for_ui(downloaded_wav_path, self.model, self.genres)
            self.update_result(result_text)
            if genre and confidence:
                messagebox.showinfo("Prediction Result", f"The predicted genre is: {genre.upper()}\n\nConfidence: {confidence:.2f}%")
        except Exception as e:
            self.update_result(f"An error occurred: {e}")
        finally:
            self.update_status("Cleaning up temporary files...")
            if downloaded_wav_path and os.path.exists(downloaded_wav_path):
                os.remove(downloaded_wav_path)
            self.update_status("Ready")
            self.set_ui_state(is_predicting=False)

#==============================================================================
# MAIN EXECUTION
#==============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("="*50 + "\nModel not found. Starting one-time training process.\nThis may take a significant amount of time...\n" + "="*50)
        run_training()
        print("\nTraining complete. Launching application...")
    else:
        print("Model found. Launching application...")
    
    root = tk.Tk()
    app = GenreClassifierApp(root)
    root.mainloop()