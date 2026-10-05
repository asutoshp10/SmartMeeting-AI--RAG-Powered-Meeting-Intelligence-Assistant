import tempfile
from pathlib import Path

import streamlit as st

from main import run_pipeline
from core.rag_engine import ask_question


st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
:root {
    --bg: #07111f;
    --panel: rgba(9, 18, 32, 0.78);
    --panel-strong: rgba(13, 25, 43, 0.92);
    --border: rgba(255, 255, 255, 0.12);
    --text: #f6f7fb;
    --muted: #aeb8cb;
    --accent: #ffb86b;
    --accent-2: #6be3ff;
    --accent-3: #9e8cff;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(107, 227, 255, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(255, 184, 107, 0.16), transparent 28%),
        radial-gradient(circle at bottom center, rgba(158, 140, 255, 0.16), transparent 32%),
        linear-gradient(180deg, #050b13 0%, #091423 45%, #07111f 100%);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 18, 31, 0.96), rgba(7, 13, 24, 0.96));
    border-right: 1px solid var(--border);
}

.hero {
    padding: 2rem 2.2rem;
    border: 1px solid var(--border);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
    box-shadow: 0 20px 80px rgba(0, 0, 0, 0.28);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.05), transparent);
    transform: translateX(-100%);
    animation: sweep 7s ease-in-out infinite;
}

@keyframes sweep {
    0%, 35% { transform: translateX(-100%); }
    60%, 100% { transform: translateX(100%); }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(107, 227, 255, 0.22);
    background: rgba(107, 227, 255, 0.08);
    color: var(--accent-2);
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 0.8rem 0 0.45rem;
    font-size: clamp(2.2rem, 5vw, 4.4rem);
    line-height: 0.95;
    letter-spacing: -0.05em;
}

.hero p {
    margin: 0;
    max-width: 58rem;
    color: var(--muted);
    font-size: 1.03rem;
    line-height: 1.7;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1.2rem;
}

.metric-card,
.glass-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    backdrop-filter: blur(14px);
    box-shadow: 0 16px 55px rgba(0, 0, 0, 0.24);
}

.metric-card {
    padding: 1rem 1.1rem;
    animation: float 6s ease-in-out infinite;
}

.metric-card:nth-child(2) { animation-delay: 0.6s; }
.metric-card:nth-child(3) { animation-delay: 1.2s; }
.metric-card:nth-child(4) { animation-delay: 1.8s; }

.metric-label {
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.11em;
}

.metric-value {
    margin-top: 0.55rem;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
}

.metric-subtext {
    margin-top: 0.35rem;
    color: var(--muted);
    font-size: 0.92rem;
}

.section-heading {
    margin: 1.5rem 0 0.85rem;
    font-size: 1.15rem;
    letter-spacing: 0.02em;
}

.glass-card {
    padding: 1.15rem 1.25rem;
}

.glass-card h3 {
    margin-top: 0;
}

.fade-in {
    animation: fadeInUp 0.65s ease both;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 999px;
    border: 1px solid transparent;
    padding: 0.4rem 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255,184,107,0.14), rgba(107,227,255,0.12));
    border-color: rgba(255, 255, 255, 0.14);
}

.stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea {
    background-color: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: var(--text) !important;
}

.stButton button {
    border: 0;
    border-radius: 999px;
    padding: 0.65rem 1.15rem;
    background: linear-gradient(135deg, #ffb86b, #ff8d6b);
    color: #111827;
    font-weight: 700;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    box-shadow: 0 10px 30px rgba(255, 144, 107, 0.22);
}

.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 34px rgba(255, 144, 107, 0.28);
}

.prompt-chip {
    display: inline-block;
    margin: 0.25rem 0.35rem 0 0;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(107, 227, 255, 0.1);
    border: 1px solid rgba(107, 227, 255, 0.15);
    color: var(--accent-2);
    font-size: 0.84rem;
}

.muted-copy {
    color: var(--muted);
}

@media (max-width: 900px) {
    .metric-grid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 640px) {
    .metric-grid { grid-template-columns: 1fr; }
    .hero { padding: 1.25rem; }
}
</style>
"""


def render_metric(label: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card fade-in">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def summarize_counts(text: str) -> str:
    if not text:
        return "0 chars"
    word_count = len(text.split())
    return f"{word_count:,} words"


def split_sources_block(content: str) -> tuple[str, str]:
    if not content or "\n\nSources:\n" not in content:
        return content, ""
    body, sources = content.split("\n\nSources:\n", 1)
    return body, sources


def render_cited_section(title: str, content: str) -> None:
    body, sources = split_sources_block(content)
    st.markdown(f"#### {title}")
    st.markdown(body)
    if sources:
        st.markdown("##### Sources")
        st.markdown(sources)


st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero fade-in">
        <div class="hero-badge">AI meeting assistant · Streamlit UI</div>
        <h1>Turn audio, video, and meetings into a clean working brief.</h1>
        <p>
            Upload a file or paste a YouTube link, then generate a title, transcript,
            summary, action items, decisions, unresolved questions, and a chat-ready
            knowledge base.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.markdown("## Controls")
    st.caption("Choose a source and run the same backend pipeline used by the CLI.")

    source_type = st.radio("Source type", ["YouTube URL", "Local file upload"], horizontal=False)
    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    source_value = ""
    uploaded_file = None
    if source_type == "YouTube URL":
        source_value = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Upload media",
            type=["mp3", "wav", "mp4", "m4a", "mov", "webm", "aac", "flac"],
        )

    run_clicked = st.button("Generate analysis", use_container_width=True)
    clear_clicked = st.button("Reset session", use_container_width=True)

    st.markdown("### Prompts")
    st.markdown('<span class="prompt-chip">What are the next actions?</span>', unsafe_allow_html=True)
    st.markdown('<span class="prompt-chip">What decisions were made?</span>', unsafe_allow_html=True)
    st.markdown('<span class="prompt-chip">Any unanswered questions?</span>', unsafe_allow_html=True)

if clear_clicked:
    st.session_state.result = None
    st.session_state.chat_history = []
    st.rerun()

if run_clicked:
    if source_type == "YouTube URL" and not source_value:
        st.error("Enter a YouTube URL before generating the analysis.")
    elif source_type == "Local file upload" and not uploaded_file:
        st.error("Upload a local media file before generating the analysis.")
    else:
        with st.spinner("Running transcription, summarization, extraction, and RAG setup..."):
            source_path = source_value
            temp_path = None
            if uploaded_file is not None:
                temp_path = save_uploaded_file(uploaded_file)
                source_path = temp_path

            try:
                result = run_pipeline(source_path, language)
                st.session_state.result = result
                st.session_state.chat_history = []
            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink(missing_ok=True)

result = st.session_state.result

if result:
    transcript = result.get("transcript", "")
    st.markdown("### Output Overview")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric("Title", result.get("title", "-"), "Generated from the transcript")
    with metric_cols[1]:
        render_metric("Transcript", summarize_counts(transcript), f"{len(transcript):,} characters")
    with metric_cols[2]:
        render_metric("Mode", language.title(), "Whisper or Sarvam routing")
    with metric_cols[3]:
        render_metric("RAG", "Ready", "Chat stays grounded in the transcript")

    tabs = st.tabs(["Summary", "Transcript", "Action Items", "Key Decisions", "Open Questions", "Chat"])

    with tabs[0]:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        render_cited_section("Summary", result.get("summary", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown("#### Transcript")
        timestamped_transcript = result.get("timestamped_transcript", "")
        if timestamped_transcript:
            st.caption("Timestamped segments are preserved for citation-aware retrieval and extraction.")
            st.text_area(
                "Timestamped transcript",
                timestamped_transcript,
                height=220,
                label_visibility="collapsed",
            )
            st.markdown("---")
            with st.expander("Show timestamped segment table", expanded=False):
                st.dataframe(
                    [
                        {
                            "start": segment.get("start"),
                            "end": segment.get("end"),
                            "text": segment.get("text", ""),
                        }
                        for segment in result.get("segments", [])
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
        st.text_area("Full transcript", transcript, height=420, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        render_cited_section("Action Items", result.get("action_items", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        render_cited_section("Key Decisions", result.get("key_decisions", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        render_cited_section("Open Questions", result.get("open_questions", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[5]:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown("#### Ask the meeting")
        st.caption("Questions are answered only from the transcript context and include cited timestamp sources.")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                body, sources = split_sources_block(message["content"])
                st.markdown(body)
                if sources:
                    st.caption("Sources")
                    st.markdown(sources)

        user_question = st.chat_input("Ask about the transcript, decisions, owners, or next steps")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            answer = ask_question(result["rag_chain"], user_question)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                body, sources = split_sources_block(answer)
                st.markdown(body)
                if sources:
                    st.caption("Sources")
                    st.markdown(sources)

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div class="glass-card fade-in">
            <h3>What this frontend supports</h3>
            <p class="muted-copy">
                Run the full pipeline from a URL or uploaded file, then explore the transcript,
                summary, action items, decisions, open questions, and the grounded RAG chat.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
