# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import re
import tempfile
import os

app = FastAPI(title="🎧 AI Music Metadata Tagger (Hybrid Audio + Text)")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---------- Load models ----------
print("🔊 Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

print("🤗 Loading text classifiers...")
mood_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3,
    framework="pt",
    model_kwargs={"use_safetensors": False}
)
genre_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt",
    model_kwargs={"use_safetensors": False}
)
possible_genres = [
    "rock", "pop", "jazz", "hip hop", "classical", "blues", "electronic",
    "folk", "reggae", "metal", "country", "ambient", "latin", "funk", "punk"
]

# ---------- Helpers ----------
def get_embedding(audio_path: str):
    """Extract mean YAMNet embedding."""
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, embeddings, _ = yamnet_model(waveform)
        return np.mean(embeddings, axis=0)
    except Exception as e:
        print("Error extracting embedding:", e)
        return np.zeros(1024)

def predict_from_text(title, artist, album):
    """Text-based predictions with confidence scores."""
    text = f"Song: {title} by {artist} from the album {album}."
    text = re.sub(r"\s+", " ", text.strip())

    mood_preds = mood_classifier(text)
    moods = [m["label"] for m in mood_preds[0]] if isinstance(mood_preds, list) else ["Unknown"]
    mood_scores = [float(m["score"]) * 100 for m in mood_preds[0]] if isinstance(mood_preds, list) else [0.0]

    genre_preds = genre_classifier(text, possible_genres)
    genres = [g for g, score in zip(genre_preds["labels"], genre_preds["scores"]) if score > 0.15]
    genre_scores = [float(score) * 100 for score in genre_preds["scores"] if score > 0.15]

    return moods[:2], mood_scores[:2], genres[:2], genre_scores[:2]

def predict_from_audio(audio_path):
    """Audio-based rough predictions from energy/brightness with confidence."""
    emb = get_embedding(audio_path)
    energy = float(np.mean(np.abs(emb)))
    brightness = float(np.mean(emb[512:]))

    # Simple heuristics with confidence based on how far from threshold
    mood = "energetic" if energy > 0.02 else "calm"
    mood_confidence = min(100, max(50, abs(energy - 0.02) * 2000 + 60))
    
    genre = "electronic" if brightness > 0 else "acoustic"
    genre_confidence = min(100, max(50, abs(brightness) * 500 + 60))
    
    return mood, mood_confidence, genre, genre_confidence

def combine_predictions(text_moods, text_mood_scores, text_genres, text_genre_scores, 
                       audio_mood, audio_mood_conf, audio_genre, audio_genre_conf):
    """Simple fusion rule-based ensemble with confidence scores."""
    # Prefer text results for genre, audio for mood balance
    final_moods = list({audio_mood, *text_moods})[:2]
    final_genres = list({audio_genre, *text_genres})[:2]
    
    # Calculate average confidence (weighted towards text predictions)
    mood_confidence = (audio_mood_conf * 0.4 + (sum(text_mood_scores[:2]) / len(text_mood_scores[:2]) if text_mood_scores else 0) * 0.6)
    genre_confidence = (audio_genre_conf * 0.3 + (sum(text_genre_scores[:2]) / len(text_genre_scores[:2]) if text_genre_scores else 0) * 0.7)
    
    return final_moods, mood_confidence, final_genres, genre_confidence

# ---------- Hybrid Endpoint ----------
@app.post("/predict_hybrid")
async def predict_hybrid(
    title: str = Form(...),
    artist: str = Form(...),
    album: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Predictions
        text_moods, text_mood_scores, text_genres, text_genre_scores = predict_from_text(title, artist, album)
        audio_mood, audio_mood_conf, audio_genre, audio_genre_conf = predict_from_audio(tmp_path)
        final_moods, mood_confidence, final_genres, genre_confidence = combine_predictions(
            text_moods, text_mood_scores, text_genres, text_genre_scores,
            audio_mood, audio_mood_conf, audio_genre, audio_genre_conf
        )

        return {
            "input": {"title": title, "artist": artist, "album": album},
            "audio_based": {
                "mood": audio_mood, 
                "mood_confidence": round(audio_mood_conf, 1),
                "genre": audio_genre,
                "genre_confidence": round(audio_genre_conf, 1)
            },
            "text_based": {
                "mood": text_moods, 
                "mood_scores": [round(s, 1) for s in text_mood_scores],
                "genre": text_genres,
                "genre_scores": [round(s, 1) for s in text_genre_scores]
            },
            "final_prediction": {
                "mood": final_moods, 
                "mood_confidence": round(mood_confidence, 1),
                "genre": final_genres,
                "genre_confidence": round(genre_confidence, 1)
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    """Serve the web UI"""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "🎶 Hybrid AI Music Tagger API running!"}

@app.get("/api/status")
def status():
    """API status endpoint"""
    return {"message": "🎶 Hybrid AI Music Tagger API running!"}
