```python
import streamlit as st
import requests
import io
import tempfile
import os
from gtts import gTTS
from groq import Groq

st.set_page_config(page_title="VoxAI", page_icon="🎙️", layout="centered")

# ------------------ SYSTEM PROMPT ------------------
SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. "
    "Reply ONLY in the user's selected language. "
    "Keep responses concise (1–3 sentences), conversational, and suitable for voice output."
)

# ------------------ LANGUAGE OPTIONS ------------------
lang_options = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh"
}

# ------------------ FILTER ------------------
JUNK_PHRASES = {
    "", "you", "the", "and", "i", "a", "an", "to", "of", "is", "it",
    "hello", "hi", "hey", "bye", "okay", "ok", "hmm", "um", "uh",
    "thank you", "thanks"
}

MIN_WORDS = 2
MIN_CHARS = 6

def is_valid_transcript(text: str) -> bool:
    clean = text.strip().lower().rstrip(".,!?")
    return not (
        clean in JUNK_PHRASES or
        len(clean) < MIN_CHARS or
        len(clean.split()) < MIN_WORDS
    )

# ------------------ AI RESPONSE ------------------
def get_ai_response(history, openrouter_key, language_name):
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + f" The user speaks {language_name}."}
            ] + history,
            "max_tokens": 300,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ------------------ TRANSCRIBE ------------------
def transcribe_audio(audio_bytes, groq_key, lang):
    client = Groq(api_key=groq_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=("recording.wav", f, "audio/wav"),
                model="whisper-large-v3-turbo",
                language=lang,
            )
        return result.text.strip()
    finally:
        os.unlink(tmp_path)

# ------------------ TTS ------------------
def tts_bytes(text, lang_code):
    tts = gTTS(text=text, lang=lang_code, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

# ------------------ UI ------------------
st.title("🎙️ VoxAI")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    selected_lang_name = st.selectbox(
        "Language",
        list(lang_options.keys())
    )
    lang = lang_options[selected_lang_name]

    st.caption(f"Selected: **{selected_lang_name}**")

    if st.button("🗑️ Clear Chat"):
        st.session_state["history"] = []
        st.rerun()

# API Keys
try:
    openrouter_key = st.secrets["OPENROUTER_API_KEY"]
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Add API keys in Streamlit secrets")
    st.stop()

# Session
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.markdown(f"🧑 {msg['content']}")
    else:
        st.markdown(f"🤖 {msg['content']}")

# ------------------ VOICE INPUT ------------------
st.markdown("### 🎤 Voice Input")
audio = st.audio_input("Record")

if audio is not None:
    audio_bytes = audio.read()

    with st.spinner("Transcribing..."):
        try:
            user_text = transcribe_audio(audio_bytes, groq_key, lang)
        except Exception as e:
            st.error(e)
            st.stop()

    if not is_valid_transcript(user_text):
        st.warning("Too short or unclear, try again.")
    else:
        st.success(f"Heard: {user_text}")

        st.session_state["history"].append({"role": "user", "content": user_text})

        with st.spinner("Thinking..."):
            ai_text = get_ai_response(
                st.session_state["history"],
                openrouter_key,
                selected_lang_name
            )

        st.session_state["history"].append({"role": "assistant", "content": ai_text})

        # Voice Output
        st.audio(tts_bytes(ai_text, lang), format="audio/mp3", autoplay=True)

        st.rerun()

# ------------------ TEXT INPUT ------------------
st.markdown("### 💬 Text Input")

col1, col2 = st.columns([5,1])

with col1:
    text = st.text_input("Type message")

with col2:
    send = st.button("Send")

if send and text.strip():
    st.session_state["history"].append({"role": "user", "content": text})

    with st.spinner("Thinking..."):
        ai_text = get_ai_response(
            st.session_state["history"],
            openrouter_key,
            selected_lang_name
        )

    st.session_state["history"].append({"role": "assistant", "content": ai_text})

    st.audio(tts_bytes(ai_text, lang), format="audio/mp3", autoplay=True)

    st.rerun()
```
