import streamlit as st
import requests
import io
import base64
import tempfile
import os
from datetime import date
from gtts import gTTS
from groq import Groq

st.set_page_config(page_title="VoxAI", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700;800&family=Nunito+Sans:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Nunito Sans', sans-serif;
    background: #f5f0eb;
    color: #2d2a26;
}
.stApp {
    background: linear-gradient(160deg, #fdf8f3 0%, #f0e9e0 50%, #e8ddd4 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #fefcf9 !important;
    border-right: 1px solid #e8e0d8 !important;
}
[data-testid="stSidebar"] * { color: #4a4540 !important; }

/* ── Header ── */
.vox-header { text-align: center; padding: 2rem 0 0.8rem; }
.vox-avatar {
    width: 68px; height: 68px;
    background: linear-gradient(135deg, #e07b4f, #d4956a);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    margin: 0 auto 1rem;
    box-shadow: 0 8px 24px rgba(224,123,79,0.3);
}
.vox-wordmark {
    font-family: 'Nunito', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #2d2a26; letter-spacing: 2px;
}
.vox-wordmark span { color: #e07b4f; }
.vox-tagline {
    font-size: 12px; letter-spacing: 3px;
    text-transform: uppercase; color: #a89880;
    margin-top: 4px; font-weight: 600;
}
.vox-line {
    width: 60px; height: 2px;
    background: linear-gradient(90deg, transparent, #e07b4f, transparent);
    margin: 1rem auto 0; border-radius: 2px;
}

/* ── Chat area ── */
.chat-wrap {
    max-height: 50vh; overflow-y: auto;
    padding: 0.5rem 0.2rem; margin-bottom: 0.5rem;
    scrollbar-width: thin; scrollbar-color: #d4c9be transparent;
}
.bubble-user {
    display: flex; justify-content: flex-end;
    margin: 10px 0; animation: fadeUp 0.25s ease;
}
.bubble-user span {
    background: linear-gradient(135deg, #e07b4f, #d4956a);
    color: #fff; padding: 12px 18px;
    border-radius: 22px 22px 5px 22px;
    max-width: 78%; font-size: 14.5px; line-height: 1.6;
    font-weight: 600; box-shadow: 0 4px 16px rgba(224,123,79,0.25);
}
.bubble-ai {
    display: flex; justify-content: flex-start;
    margin: 10px 0; align-items: flex-end; gap: 8px;
    animation: fadeUp 0.25s ease;
}
.ai-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #6b9fd4, #5b8fc4);
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 15px; flex-shrink: 0;
    box-shadow: 0 3px 10px rgba(107,159,212,0.3);
}
.bubble-ai span {
    background: #fff; border: 1px solid #e8e0d8; color: #2d2a26;
    padding: 12px 18px; border-radius: 22px 22px 22px 5px;
    max-width: 75%; font-size: 14.5px; line-height: 1.6;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.empty-state { text-align: center; padding: 3rem 1rem 2rem; color: #b8a898; }
.empty-icon  { font-size: 3rem; margin-bottom: 0.8rem; }
.empty-text  { font-size: 14px; font-weight: 700; letter-spacing: 0.5px; }
.empty-sub   { font-size: 12px; margin-top: 4px; color: #c8b8a8; }

/* ── Transcript ── */
.transcript-box {
    background: #fff8f2; border: 1.5px solid #f0d8c8;
    border-left: 4px solid #e07b4f; border-radius: 12px;
    padding: 10px 16px; font-size: 13px; color: #7a6a58;
    margin: 8px 0 12px; font-style: italic;
}

/* ── Divider ── */
.vox-divider { border: none; border-top: 1.5px solid #e8e0d8; margin: 1rem 0 0.8rem; }

/* ── Text input ── */
.stTextInput > div > div > input {
    background: #fff !important;
    border: 1.5px solid #e0d8d0 !important;
    border-radius: 14px !important; color: #2d2a26 !important;
    font-family: 'Nunito Sans', sans-serif !important;
    font-size: 14px !important; padding: 10px 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #e07b4f !important;
    box-shadow: 0 0 0 3px rgba(224,123,79,0.12) !important;
}

/* ── Audio mic ── */
div[data-testid="stAudioInput"] {
    background: transparent !important; border: none !important; padding: 0 !important;
}
div[data-testid="stAudioInput"] > label { display: none !important; }
div[data-testid="stAudioInput"] button {
    background: #fff !important; border: 2px solid #e07b4f !important;
    border-radius: 50% !important; width: 52px !important; height: 52px !important;
    color: #e07b4f !important; font-size: 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(224,123,79,0.2) !important; cursor: pointer !important;
}
div[data-testid="stAudioInput"] button:hover {
    background: #e07b4f !important; color: #fff !important;
    box-shadow: 0 6px 20px rgba(224,123,79,0.35) !important; transform: scale(1.05) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e07b4f, #d4956a) !important;
    color: #fff !important; border: none !important; border-radius: 14px !important;
    font-family: 'Nunito', sans-serif !important; font-size: 14px !important;
    font-weight: 700 !important; height: 48px !important; padding: 0 22px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(224,123,79,0.3) !important; white-space: nowrap !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(224,123,79,0.4) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #f5ede4 !important; color: #a08060 !important;
    box-shadow: none !important; border: 1px solid #e0d0c0 !important;
}

/* ── Speaking badge ── */
.speaking-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #fff0e8; border: 1.5px solid #f0d0b8;
    border-radius: 20px; padding: 5px 14px;
    font-size: 12px; color: #e07b4f; font-weight: 700;
    letter-spacing: 0.5px; margin: 6px 0;
}
.pulse-dot {
    width: 8px; height: 8px; background: #e07b4f;
    border-radius: 50%; animation: pulse 1s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

/* ── Footer ── */
.vox-footer {
    text-align: center; font-size: 10px; color: #c8b8a8;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 1.2rem 0 0.5rem; font-weight: 600;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Live date in system prompt ────────────────────────────────────────────────
today_str = date.today().strftime("%B %d, %Y")
SYSTEM_PROMPT = (
    f"You are a helpful, friendly voice assistant. "
    f"Today's date is {today_str}. "
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

LANGUAGES = {
    "English":    "en",
    "Hindi":      "hi",
    "Telugu":     "te",
    "Tamil":      "ta",
    "French":     "fr",
    "Spanish":    "es",
    "German":     "de",
    "Japanese":   "ja",
    "Korean":     "ko",
    "Chinese":    "zh",
    "Arabic":     "ar",
    "Portuguese": "pt",
    "Russian":    "ru",
    "Bengali":    "bn",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_valid_transcript(text: str) -> bool:
    clean = text.strip().lower().rstrip(".,!?")
    if clean in JUNK_PHRASES:
        return False
    if len(clean) < MIN_CHARS or len(clean.split()) < MIN_WORDS:
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


def transcribe_audio(audio_bytes: bytes, groq_key: str, lang_code: str) -> str:
    client = Groq(api_key=groq_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=("recording.wav", f, "audio/wav"),
                model="whisper-large-v3-turbo",
                language=lang_code,
                prompt="User is asking a question or giving a command.",
            )
        return result.text.strip()
    finally:
        os.unlink(tmp_path)


def tts_autoplay(text: str, lang_code: str):
    """Embed TTS audio as base64 in a hidden <audio autoplay> tag — works on Streamlit Cloud."""
    tts = gTTS(text=text, lang=lang_code, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <div class="speaking-badge">
        <div class="pulse-dot"></div> Speaking…
    </div>
    """, unsafe_allow_html=True)


# ── API keys ──────────────────────────────────────────────────────────────────
try:
    openrouter_key = st.secrets["OPENROUTER_API_KEY"]
    groq_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    openrouter_key = ""
    groq_key = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎙️ VoxAI")
    st.markdown("---")
    if openrouter_key and groq_key:
        st.success("🔑 API keys loaded")
    else:
        st.warning("⚠️ API keys missing")

    st.markdown("**🌐 Language**")
    lang_name = st.selectbox(
        "Language", list(LANGUAGES.keys()), index=0, label_visibility="collapsed"
    )
    lang_code = LANGUAGES[lang_name]
    st.caption(f"🗣️ Speaking in: **{lang_name}**")

    st.markdown("---")
    st.markdown(f"📅 **Today:** {today_str}")
    st.markdown("---")

    if st.button("🗑️ Clear conversation"):
        st.session_state["history"] = []
        st.session_state["pending_tts"] = None
        st.rerun()

    st.markdown(
        "<div style='font-size:10px;color:#b8a898;margin-top:2rem;text-align:center;letter-spacing:1px;'>Groq · OpenRouter · gTTS</div>",
        unsafe_allow_html=True,
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "pending_tts" not in st.session_state:
    st.session_state["pending_tts"] = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vox-header">
    <div class="vox-avatar">🎙️</div>
    <div class="vox-wordmark">Vox<span>AI</span></div>
    <div class="vox-tagline">Your personal voice assistant</div>
    <div class="vox-line"></div>
</div>
""", unsafe_allow_html=True)

# ── API key gate ──────────────────────────────────────────────────────────────
if not openrouter_key or not groq_key:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">⚠️</div>
        <div style="font-size:15px;line-height:2;color:#7a6a58;">
            Add your API keys in<br>
            <strong style="color:#e07b4f">Streamlit Cloud → App Settings → Secrets</strong><br><br>
            <code style="background:#fff;border:1.5px solid #e8d8c8;padding:12px 20px;
                  border-radius:12px;font-size:13px;color:#e07b4f;display:inline-block;line-height:2;">
            OPENROUTER_API_KEY = "sk-or-v1-..."<br>
            GROQ_API_KEY = "gsk_..."
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Play pending TTS after rerun ──────────────────────────────────────────────
if st.session_state["pending_tts"]:
    tts_autoplay(st.session_state["pending_tts"], lang_code)
    st.session_state["pending_tts"] = None

# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state["history"]:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state["history"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user"><span>{msg["content"]}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bubble-ai"><div class="ai-icon">🤖</div><span>{msg["content"]}</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">💬</div>
        <div class="empty-text">Hi there! I'm VoxAI</div>
        <div class="empty-sub">Tap the mic or type below — I'll talk back to you!</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="vox-divider">', unsafe_allow_html=True)

# ── Input bar: mic | text input | send ───────────────────────────────────────
col_mic, col_text, col_send = st.columns([1, 5, 1.3])

with col_mic:
    audio_value = st.audio_input("🎤", label_visibility="collapsed")

with col_text:
    text_input = st.text_input(
        "msg", placeholder="Type your message here...",
        label_visibility="collapsed", key="text_msg",
    )

with col_send:
    send_btn = st.button("Send ➤")

# ── Handle voice ──────────────────────────────────────────────────────────────
if audio_value is not None:
    audio_bytes = audio_value.read()
    with st.spinner("🎧 Listening..."):
        try:
            user_text = transcribe_audio(audio_bytes, groq_key, lang_code)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

    if not user_text or not is_valid_transcript(user_text):
        st.warning(f'Heard: **"{user_text}"** — too short or unclear. Please try again.')
    else:
        st.markdown(
            f'<div class="transcript-box">📝 Heard: <b>"{user_text}"</b></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Send this", key="confirm_send"):
                st.session_state["history"].append({"role": "user", "content": user_text})
                with st.spinner("🤔 Thinking..."):
                    try:
                        ai_text = get_ai_response(st.session_state["history"], openrouter_key)
                        st.session_state["history"].append({"role": "assistant", "content": ai_text})
                        st.session_state["pending_tts"] = ai_text
                    except Exception as e:
                        st.error(f"AI error: {e}")
                st.rerun()
        with c2:
            if st.button("🔄 Re-record", key="rerecord"):
                st.rerun()

# ── Handle text send ──────────────────────────────────────────────────────────
if send_btn and text_input.strip():
    st.session_state["history"].append({"role": "user", "content": text_input.strip()})
    with st.spinner("🤔 Thinking..."):
        try:
            ai_text = get_ai_response(st.session_state["history"], openrouter_key)
            st.session_state["history"].append({"role": "assistant", "content": ai_text})
            st.session_state["pending_tts"] = ai_text
        except Exception as e:
            st.error(f"Error: {e}")
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="vox-footer">Groq Whisper · OpenRouter GPT-4o-mini · gTTS · Streamlit</div>',
    unsafe_allow_html=True,
)
