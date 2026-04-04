# ── Cached emotion classifier (loaded ONCE, reused forever) ──────────────────
@st.cache_resource(show_spinner=False)
def load_emotion_classifier():
    """Load SpeechBrain classifier once and cache it for the session."""
    try:
        from speechbrain.inference.interfaces import foreign_class
        classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir="pretrained_models/emotion-recognition",
        )
        return classifier, None
    except ImportError:
        return None, "missing"
    except Exception as e:
        return None, str(e)


def detect_emotion_fast_fallback(audio_bytes: bytes) -> dict:
    """
    Fast acoustic-feature emotion detection using librosa.
    No model download — runs in under 0.5 seconds.
    """
    try:
        import librosa
        import numpy as np

        buf = io.BytesIO(audio_bytes)
        y, sr = librosa.load(buf, sr=16000, mono=True, duration=10.0)

        if len(y) < sr * 0.3:
            return {**EMOTION_CONFIG["neutral"], "confidence": 50.0, "_fallback": True}

        rms        = float(np.mean(librosa.feature.rms(y=y)))
        f0, _, _   = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        f0_vals    = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        pitch_mean = float(np.mean(f0_vals)) if len(f0_vals) > 0 else 0.0
        pitch_std  = float(np.std(f0_vals))  if len(f0_vals) > 0 else 0.0
        zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        energy_norm = min(rms / 0.15, 1.0)

        if energy_norm > 0.75 and pitch_std > 40:
            emotion_key, conf = "angry",    70.0
        elif energy_norm > 0.55 and pitch_mean > 220 and pitch_std > 30:
            emotion_key, conf = "happy",    68.0
        elif energy_norm < 0.25 and pitch_mean < 140 and pitch_std < 15:
            emotion_key, conf = "sad",      65.0
        elif energy_norm > 0.6 and zcr > 0.08 and pitch_std > 50:
            emotion_key, conf = "surprise", 62.0
        elif energy_norm < 0.2 and zcr < 0.04:
            emotion_key, conf = "fear",     58.0
        elif 0.25 <= energy_norm <= 0.55 and pitch_std < 25:
            emotion_key, conf = "neutral",  72.0
        else:
            emotion_key, conf = "neutral",  55.0

        cfg = EMOTION_CONFIG.get(emotion_key, EMOTION_CONFIG["neutral"])
        return {**cfg, "confidence": conf, "_fallback": True}

    except Exception:
        return {**EMOTION_CONFIG["neutral"], "confidence": 0.0, "_fallback": True}


def detect_emotion(audio_bytes: bytes) -> dict:
    """
    Priority chain:
    1. Cached SpeechBrain model  → fast after first load (~1-2s)
    2. Acoustic librosa fallback → always fast (~0.5s), no model needed
    """
    classifier, err = load_emotion_classifier()

    if err == "missing":
        result = detect_emotion_fast_fallback(audio_bytes)
        result["_missing"] = True
        return result

    if classifier is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            out_prob, score, index, text_lab = classifier.classify_file(tmp_path)
            emotion_raw = text_lab[0].lower()
            label_map = {
                "ang": "angry",  "angry":    "angry",
                "hap": "happy",  "happy":    "happy",  "exc": "happy",
                "sad": "sad",
                "neu": "neutral","neutral":  "neutral",
                "fea": "fear",   "fear":     "fear",
                "dis": "disgust","disgust":  "disgust",
                "sur": "surprise","surprise":"surprise",
            }
            emotion_key = label_map.get(emotion_raw, "neutral")
            confidence  = round(float(score[0]) * 100, 1)
            cfg = EMOTION_CONFIG.get(emotion_key, EMOTION_CONFIG["neutral"])
            return {**cfg, "confidence": confidence}
        except Exception:
            pass
        finally:
            os.unlink(tmp_path)

    return detect_emotion_fast_fallback(audio_bytes)


def render_emotion_badge(emotion: dict):
    conf_str = f" · {emotion.get('confidence', 0):.0f}%" if emotion.get("confidence", 0) > 0 else ""
    if emotion.get("_missing"):
        source_note = " · acoustic"
        st.markdown(
            f'<div class="emotion-badge {emotion["css"]}">'
            f'{emotion["emoji"]} Tone: <strong>{emotion["label"]}</strong>{conf_str}{source_note}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption("💡 Install `speechbrain` for deeper AI emotion analysis.")
        return
    source_note = " · acoustic" if emotion.get("_fallback") else " · wav2vec2"
    st.markdown(
        f'<div class="emotion-badge {emotion["css"]}">'
        f'{emotion["emoji"]} Tone: <strong>{emotion["label"]}</strong>{conf_str}{source_note}'
        f'</div>',
        unsafe_allow_html=True,
    )
