# app.py
from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import re
import tempfile

app = FastAPI(title="🎧 AI Music Metadata Tagger (Hybrid Audio + Text)")

# ---------- Load models ----------
print("🔊 Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

print("🤗 Loading text classifiers...")
mood_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3
)
genre_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
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
    """Text-based predictions."""
    text = f"Song: {title} by {artist} from the album {album}."
    text = re.sub(r"\s+", " ", text.strip())

    mood_preds = mood_classifier(text)
    moods = [m["label"] for m in mood_preds[0]] if isinstance(mood_preds, list) else ["Unknown"]

    genre_preds = genre_classifier(text, possible_genres)
    genres = [g for g, score in zip(genre_preds["labels"], genre_preds["scores"]) if score > 0.15]

    return moods[:2], genres[:2]

def predict_from_audio(audio_path):
    """Audio-based rough predictions from energy/brightness."""
    emb = get_embedding(audio_path)
    energy = float(np.mean(np.abs(emb)))
    brightness = float(np.mean(emb[512:]))

    # Simple heuristics
    mood = "energetic" if energy > 0.02 else "calm"
    genre = "electronic" if brightness > 0 else "acoustic"
    return mood, genre

def combine_predictions(text_moods, text_genres, audio_mood, audio_genre):
    """Simple fusion rule-based ensemble."""
    # Prefer text results for genre, audio for mood balance
    final_moods = list({audio_mood, *text_moods})[:2]
    final_genres = list({audio_genre, *text_genres})[:2]
    return final_moods, final_genres

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
        text_moods, text_genres = predict_from_text(title, artist, album)
        audio_mood, audio_genre = predict_from_audio(tmp_path)
        final_moods, final_genres = combine_predictions(text_moods, text_genres, audio_mood, audio_genre)

        return {
            "input": {"title": title, "artist": artist, "album": album},
            "audio_based": {"mood": audio_mood, "genre": audio_genre},
            "text_based": {"mood": text_moods, "genre": text_genres},
            "final_prediction": {"mood": final_moods, "genre": final_genres}
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "🎶 Hybrid AI Music Tagger API running!"}
