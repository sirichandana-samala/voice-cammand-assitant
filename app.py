import streamlit as st
import requests
import base64
import io
import tempfile
import os
from gtts import gTTS
from groq import Groq

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="VoxAI", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080810;
    color: #f0f0f5;
}
.stApp { background-color: #080810; }

.vox-header { text-align: center; padding: 2.5rem 0 1.5rem; }
.vox-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem; font-weight: 700; letter-spacing: 6px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.vox-sub {
    font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    color: rgba(255,255,255,0.25); margin-top: 4px;
}
.vox-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.5rem 0; }
.section-label {
    font-size: 10px; letter-spacing: 2.5px; text-transform: uppercase;
    color: rgba(255,255,255,0.3); margin-bottom: 0.5rem; font-weight: 600;
}
.bubble-user { display: flex; justify-content: flex-end; margin: 8px 0; }
.bubble-user span {
    background: linear-gradient(135deg, #7c3aed, #4f46e5); color: #fff;
    padding: 10px 18px; border-radius: 20px 20px 4px 20px;
    max-width: 78%; font-size: 14px; line-height: 1.6;
    box-shadow: 0 4px 20px rgba(124,58,237,0.25);
}
.bubble-ai { display: flex; justify-content: flex-start; margin: 8px 0; }
.bubble-ai span {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
    color: #e2e8f0; padding: 10px 18px; border-radius: 20px 20px 20px 4px;
    max-width: 78%; font-size: 14px; line-height: 1.6;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-size: 13px !important;
    font-weight: 600 !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; border: none !important; }
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #f0f0f5 !important;
    font-family: 'Syne', sans-serif !important; font-size: 14px !important;
}
div[data-testid="stAudioInput"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 14px !important; padding: 12px !important;
}
.vox-footer {
    text-align: center; font-size: 10px; color: rgba(255,255,255,0.15);
    letter-spacing: 2px; text-transform: uppercase; padding: 2rem 0 1rem;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. "
    "Keep responses concise and conversational — ideally 1–3 sentences — "
    "since they will be read aloud. Be warm, helpful, and direct."
)

def get_ai_response(history: list, openrouter_key: str) -> str:
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://voxai.streamlit.app",
            "X-Title": "VoxAI",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
            "max_tokens": 300,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def transcribe_audio(audio_bytes: bytes, groq_key: str) -> str:
    client = Groq(api_key=groq_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=("recording.wav", f, "audio/wav"),
                model="whisper-large-v3-turbo",
            )
        return result.text.strip()
    finally:
        os.unlink(tmp_path)


def tts_bytes(text: str) -> bytes:
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="vox-header">
    <div class="vox-title">VOXAI</div>
    <div class="vox-sub">Voice · Intelligence · Conversation</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load API keys from Streamlit Secrets
# ─────────────────────────────────────────────
try:
    openrouter_key = st.secrets["OPENROUTER_API_KEY"]
    groq_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    openrouter_key = ""
    groq_key = ""

# ─────────────────────────────────────────────
# Sidebar — just clear button, keys come from secrets
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.success("🔑 API keys loaded from Secrets" if openrouter_key and groq_key else "⚠️ API keys missing in Secrets")
    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state["history"] = []
        st.rerun()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# ─────────────────────────────────────────────
# Gate on keys
# ─────────────────────────────────────────────
if not openrouter_key or not groq_key:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:rgba(255,255,255,0.35);">
        <div style="font-size:2.5rem; margin-bottom:1rem;">⚠️</div>
        <div style="font-size:14px; line-height:1.8;">
            API keys not found. Add them in<br>
            <strong style="color:rgba(255,255,255,0.6)">Streamlit Cloud → App Settings → Secrets</strong><br><br>
            <code style="background:rgba(255,255,255,0.08);padding:8px 14px;border-radius:8px;font-size:13px;">
            OPENROUTER_API_KEY = "sk-or-v1-..."<br>
            GROQ_API_KEY = "gsk_..."
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────
if st.session_state["history"]:
    for msg in st.session_state["history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user"><span>🧑 {msg["content"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bubble-ai"><span>🤖 {msg["content"]}</span></div>', unsafe_allow_html=True)
    st.markdown('<hr class="vox-divider">', unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center;padding:2rem 0;color:rgba(255,255,255,0.2);font-size:13px;">
        Say something or type below to begin...
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Voice input
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">🎤 Voice input</div>', unsafe_allow_html=True)
audio_value = st.audio_input("Record", label_visibility="collapsed")

if audio_value is not None:
    with st.spinner("Transcribing..."):
        try:
            user_text = transcribe_audio(audio_value.read(), groq_key)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

    if not user_text:
        st.warning("🔇 No speech detected. Please try again.")
    else:
        st.session_state["history"].append({"role": "user", "content": user_text})
        with st.spinner("Thinking..."):
            try:
                ai_text = get_ai_response(st.session_state["history"], openrouter_key)
                st.session_state["history"].append({"role": "assistant", "content": ai_text})
                st.audio(tts_bytes(ai_text), format="audio/mp3", autoplay=True)
            except Exception as e:
                st.error(f"AI error: {e}")
        st.rerun()

# ─────────────────────────────────────────────
# Text input
# ─────────────────────────────────────────────
st.markdown('<hr class="vox-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">✍️ Or type your message</div>', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])
with col1:
    text_input = st.text_input("msg", placeholder="Ask me anything...", label_visibility="collapsed")
with col2:
    send = st.button("Send")

if send and text_input.strip():
    st.session_state["history"].append({"role": "user", "content": text_input.strip()})
    with st.spinner("Thinking..."):
        try:
            ai_text = get_ai_response(st.session_state["history"], openrouter_key)
            st.session_state["history"].append({"role": "assistant", "content": ai_text})
            st.audio(tts_bytes(ai_text), format="audio/mp3", autoplay=True)
        except Exception as e:
            st.error(f"Error: {e}")
    st.rerun()

st.markdown('<div class="vox-footer">Groq Whisper · OpenRouter · gTTS · Streamlit</div>', unsafe_allow_html=True)