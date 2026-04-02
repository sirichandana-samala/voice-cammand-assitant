import streamlit as st
import requests
import io
import tempfile
import os
from gtts import gTTS
from groq import Groq
st.set_page_config(page_title="VoxAI", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #FFFFFF;
    color: #1a1a2e;
}
.stApp { background-color: #FFFFFF; }

.vox-header { text-align: center; padding: 2.5rem 0 1.5rem; }
.vox-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem; font-weight: 700; letter-spacing: 6px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.vox-sub {
    font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    color: rgba(0,0,0,0.35); margin-top: 4px;
}
.vox-divider { border: none; border-top: 1px solid rgba(0,0,0,0.1); margin: 1.5rem 0; }
.section-label {
    font-size: 10px; letter-spacing: 2.5px; text-transform: uppercase;
    color: rgba(0,0,0,0.4); margin-bottom: 0.5rem; font-weight: 600;
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
    background: #f1f5f9; border: 1px solid rgba(0,0,0,0.08);
    color: #1e293b; padding: 10px 18px; border-radius: 20px 20px 20px 4px;
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
    background: #f8fafc !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important; color: #1a1a2e !important;
    font-family: 'Syne', sans-serif !important; font-size: 14px !important;
}
div[data-testid="stAudioInput"] {
    background: #f8fafc !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 14px !important; padding: 12px !important;
}
.tip-box {
    background: rgba(124,58,237,0.06);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 10px; padding: 10px 16px;
    font-size: 12px; color: rgba(0,0,0,0.55);
    margin-bottom: 1rem; line-height: 1.7;
}
.transcript-preview {
    background: #f1f5f9;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 10px; padding: 12px 16px;
    font-size: 13px; color: rgba(0,0,0,0.75);
    margin: 8px 0 12px; font-style: italic;
}
.vox-footer {
    text-align: center; font-size: 10px; color: rgba(0,0,0,0.3);
    letter-spacing: 2px; text-transform: uppercase; padding: 2rem 0 1rem;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. "
    "Keep responses concise and conversational — ideally 1–3 sentences — "
    "since they will be read aloud. Be warm, helpful, and direct."
)
JUNK_PHRASES = {
    "", "you", "the", "and", "i", "a", "an", "to", "of", "is", "it",
    "hello", "hi", "hey", "bye", "okay", "ok", "hmm", "um", "uh",
    "thank you", "thanks", "hello.", "hi.", "hey.", "bye.", "ok.",
}

MIN_WORDS = 2         
MIN_CHARS = 6 
def is_valid_transcript(text: str) -> bool:
    """Filter out Whisper hallucinations and noise."""
    clean = text.strip().lower().rstrip(".,!?")
    if clean in JUNK_PHRASES:
        return False
    if len(clean) < MIN_CHARS:
        return False
    if len(clean.split()) < MIN_WORDS:
        return False
    return True


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
                language="en",           
                prompt="User is speaking a clear command or question in English.",  # ← context hint
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
st.markdown("""
<div class="vox-header">
    <div class="vox-title">VOXAI</div>
    <div class="vox-sub">Voice · Intelligence · Conversation</div>
</div>
""", unsafe_allow_html=True)
try:
    openrouter_key = st.secrets["OPENROUTER_API_KEY"]
    groq_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    openrouter_key = ""
    groq_key = ""
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.success("🔑 API keys loaded" if openrouter_key and groq_key else "⚠️ API keys missing in Secrets")
    st.markdown("---")
    st.markdown("*🌐 Language*")
    lang = st.selectbox("Language", ["en", "hi", "te", "ta", "fr", "es", "de", "ja", "ko", "zh"],
                        label_visibility="collapsed")
    lang_names = {"en":"English","hi":"Hindi","te":"Telugu","ta":"Tamil",
                  "fr":"French","es":"Spanish","de":"German","ja":"Japanese",
                  "ko":"Korean","zh":"Chinese"}
    st.caption(f"Transcribing in: *{lang_names.get(lang, lang)}*")
    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state["history"] = []
        st.rerun()
if "history" not in st.session_state:
    st.session_state["history"] = []
if "pending_audio" not in st.session_state:
    st.session_state["pending_audio"] = None
if not openrouter_key or not groq_key:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:rgba(0,0,0,0.4);">
        <div style="font-size:2.5rem; margin-bottom:1rem;">⚠️</div>
        <div style="font-size:14px; line-height:1.8;">
            API keys not found. Add them in<br>
            <strong style="color:#4f46e5">Streamlit Cloud → App Settings → Secrets</strong><br><br>
            <code style="background:#f1f5f9;padding:8px 14px;border-radius:8px;font-size:13px;">
            OPENROUTER_API_KEY = "sk-or-v1-..."<br>
            GROQ_API_KEY = "gsk_..."
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
if st.session_state["history"]:
    for msg in st.session_state["history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user"><span>🧑 {msg["content"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bubble-ai"><span>🤖 {msg["content"]}</span></div>', unsafe_allow_html=True)
    st.markdown('<hr class="vox-divider">', unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center;padding:2rem 0;color:rgba(0,0,0,0.3);font-size:13px;">
        Say something or type below to begin...
    </div>""", unsafe_allow_html=True)
st.markdown('<div class="section-label">🎤 Voice input</div>', unsafe_allow_html=True)

st.markdown("""
<div class="tip-box">
   <b>Click mic</b> to start &nbsp;→&nbsp; speak your message &nbsp;→&nbsp; <b>click mic again</b> to stop<br>
   Wait for the waveform to appear before speaking. Speak clearly for best results.
</div>
""", unsafe_allow_html=True)

audio_value = st.audio_input("Record", label_visibility="collapsed")

if audio_value is not None:
    audio_bytes = audio_value.read()

    with st.spinner("Transcribing..."):
        try:
            # Use selected language from sidebar
            client = Groq(api_key=groq_key)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=("recording.wav", f, "audio/wav"),
                    model="whisper-large-v3-turbo",
                    language=lang,
                    prompt="User is asking a question or giving a command.",
                )
            os.unlink(tmp_path)
            user_text = result.text.strip()
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()
    if not user_text or not is_valid_transcript(user_text):
        st.warning(f" Heard: *\"{user_text}\"* — too short or unclear. Please try again with a full sentence.")
    else:
        st.markdown(f'<div class="transcript-preview">📝 Heard: <b>"{user_text}"</b></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Send this", key="confirm_send"):
                st.session_state["history"].append({"role": "user", "content": user_text})
                with st.spinner("Thinking..."):
                    try:
                        ai_text = get_ai_response(st.session_state["history"], openrouter_key)
                        st.session_state["history"].append({"role": "assistant", "content": ai_text})
                        st.audio(tts_bytes(ai_text), format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.error(f"AI error: {e}")
                st.rerun()
        with col2:
            if st.button("🔄 Re-record", key="rerecord"):
                st.rerun()

st.markdown('<hr class="vox-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label"> Or type your message</div>', unsafe_allow_html=True)
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
