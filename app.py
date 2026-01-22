import os
import time
import fitz  # PyMuPDF
import streamlit as st
from typing import List
from llama_cpp import Llama
from prompts import CHUNK_PROMPT, FINAL_PROMPT

# ==== Environment ====
MODEL_PATH = os.environ.get("MODEL_PATH", "models\model.gguf")
N_CTX = int(os.environ.get("N_CTX", "8192"))
N_BATCH = int(os.environ.get("N_BATCH", "512"))
N_THREADS = int(os.environ.get("N_THREADS", "0"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))

# Sampling
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
TOP_K = int(os.environ.get("TOP_K", "40"))
REPEAT_PENALTY = float(os.environ.get("REPEAT_PENALTY", "1.15"))

# Token budgets
MAX_TOKENS_INTERMEDIATE = int(os.environ.get("MAX_TOKENS_INTERMEDIATE", "350"))  # concise bullets
MAX_TOKENS_FINAL = int(os.environ.get("MAX_TOKENS_FINAL", "1200"))               # ≥500 words

# Chunking
MAX_CHARS = int(os.environ.get("MAX_CHARS", "5000"))  # biraz küçültmek akışı iyileştirir
OVERLAP = int(os.environ.get("OVERLAP", "400"))

# Stop dizileri
STOP_DEFAULT = ["</s>"]
STOP_FINAL = ["</s>", "\n[Chunk", "\nChunk", "Bullet summary:", "Passage:", "Q:", "Question:", "What are"]

# ==================== LLM ====================

@st.cache_resource(show_spinner=False)
def load_llm() -> Llama:
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_threads=N_THREADS if N_THREADS > 0 else None,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )

def _complete(llm: Llama, prompt: str, max_tokens: int, stop: List[str]) -> str:
    """Sadece düz completion kullan (chat-template sızıntısını önler)."""
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repeat_penalty=REPEAT_PENALTY,
        stop=stop,
        echo=False,
    )
    return out["choices"][0]["text"].strip()

def infer(llm: Llama, prompt: str, max_tokens: int, stop: List[str]) -> str:
    text = _complete(llm, prompt, max_tokens, stop).strip()
    if not text:
        retry = (
            "Write a detailed academic English summary (at least 500 words). "
            "Return only plain paragraphs (no lists, no headings, no questions).\n\n" + prompt
        )
        text = _complete(llm, retry, max_tokens, stop).strip()
    return text

# ==================== PDF utils ====================

def pdf_to_text(file_bytes: bytes) -> str:
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            pages.append(p.get_text())
    return "\n".join(pages).strip()

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Kelime ortasında kesmeyi azalt: pencerenin son %15'inde son boşluğu bul."""
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            window_start = max(i + int(max_chars * 0.85), i)
            cut = text.rfind(" ", window_start, end)
            if cut != -1 and cut > i:
                end = cut
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        i = max(0, end - overlap)
    return chunks

def clean_output(text: str) -> str:
    """Model çıktılarını süpür: Chunk etiketleri, Q&A vb. kaldır."""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if s.startswith("[Chunk"):
            continue
        if s.lower().startswith(("q:", "question:", "what are", "how can i")):
            continue
        if s.startswith("Bullet summary:") or s.startswith("Passage:"):
            continue
        lines.append(ln)
    cleaned = "\n".join(lines).strip()
    return cleaned

def summarize_chunk(llm: Llama, chunk: str) -> str:
    prompt = CHUNK_PROMPT.format(text=chunk)
    return infer(llm, prompt, MAX_TOKENS_INTERMEDIATE, STOP_DEFAULT)

def summarize_document(llm: Llama, full_text: str, max_tokens_final: int | None = None) -> str:
    """
    Final özet uzunluğu için global sabite dokunmadan, isteğe bağlı max_tokens_final alır.
    """
    # 1) Ara özetler
    if len(full_text) <= MAX_CHARS:
        stitched = f"[Chunk 1]\n{summarize_chunk(llm, full_text)}"
    else:
        chunks = chunk_text(full_text, MAX_CHARS, OVERLAP)
        st.info(f"Text split into {len(chunks)} chunks. Creating intermediate summaries…")
        partials = []
        for i, ch in enumerate(chunks, 1):
            with st.status(f"Summarizing chunk {i}/{len(chunks)}…", expanded=False):
                partials.append(f"[Chunk {i}]\n{summarize_chunk(llm, ch)}")
                time.sleep(0.05)
        stitched = "\n\n".join(partials)

    # 2) Final sentez (yalnızca düz paragraf)
    final_prompt = FINAL_PROMPT.format(text=stitched)
    tokens_budget = max_tokens_final if max_tokens_final is not None else MAX_TOKENS_FINAL
    summary = infer(llm, final_prompt, tokens_budget, STOP_FINAL)
    return clean_output(summary)

# ==================== UI ====================

st.set_page_config(page_title="PDF Summarizer (GGUF)", page_icon="🦋", layout="wide")

# State
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []   # [{"role":"user"/"assistant","content":"..."}]
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Header
st.markdown("# 🦋 PDF Summarizer")
st.markdown('<div class="small-subtitle">Light & sweet UI • now with Chat</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Model & Settings")
    st.text(f"MODEL_PATH: {MODEL_PATH}")
    st.text(f"N_CTX: {N_CTX}   |   N_GPU_LAYERS: {N_GPU_LAYERS}")
    st.text(f"MAX_CHARS: {MAX_CHARS}   |   OVERLAP: {OVERLAP}")
    st.text(f"TEMP: {TEMPERATURE}   |   TOP_P: {TOP_P}   |   TOP_K: {TOP_K}")
    st.text(f"REPEAT_PENALTY: {REPEAT_PENALTY}")
    st.text(f"MAX_TOKENS_INTERMEDIATE: {MAX_TOKENS_INTERMEDIATE}")
    st.text(f"MAX_TOKENS_FINAL: {MAX_TOKENS_FINAL}")
    st.text(f"GPU Offload: {'ON' if N_GPU_LAYERS > 0 else 'OFF'}")
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_messages = []

# Tabs
tab_chat, tab_classic = st.tabs(["💬 Chat", "🧩 Classic UI"])

# --------------- CHAT MODE ---------------
with tab_chat:
    st.markdown("### Upload & Chat")
    up = st.file_uploader(
        "Upload a PDF for chat", type=["pdf"],
        key="chat_uploader", help="Drag & drop or pick a PDF"
    )
    if up:
        raw = up.read()
        with st.spinner("Extracting text from PDF…"):
            st.session_state.pdf_text = pdf_to_text(raw)
            st.session_state.pdf_name = up.name
        st.success(f"Loaded **{st.session_state.pdf_name}** — {len(st.session_state.pdf_text):,} chars")

    # geçmişi göster
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # komut girişi
    prompt = st.chat_input(placeholder="e.g., summarize • bullet points • long summary • summary in Turkish")
    if prompt:
        # kullanıcı mesajını yaz
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # cevap alanı
        with st.chat_message("assistant"):
            if not st.session_state.get("pdf_text"):
                msg = "Önce PDF yükle, sonra komut gönder 😊"
                st.session_state.chat_messages.append({"role": "assistant", "content": msg})
                st.write(msg)
            else:
                # basit komut yorumlama
                cmd = prompt.strip().lower()

                # final özet token bütçesini yerel belirle (global'e dokunmadan)
                if "bullet" in cmd or "madde" in cmd:
                    max_final = min(MAX_TOKENS_FINAL, 600)
                elif "long" in cmd or "detailed" in cmd or "detay" in cmd:
                    max_final = max(900, MAX_TOKENS_FINAL)
                else:
                    max_final = min(MAX_TOKENS_FINAL, 800)

                # modeli yükle ve özetle
                with st.spinner("Thinking…"):
                    llm = load_llm()
                    summary = summarize_document(llm, st.session_state.pdf_text, max_tokens_final=max_final)

                # dil isteği: Türkçe ise hızlı çeviri (bullet yapısını koru)
                if "turkish" in cmd or "türkçe" in cmd:
                    llm = load_llm()
                    tr = _complete(
                        llm,
                        f"Translate the following to natural Turkish, keep bullet structure if any:\n\n{summary}",
                        max_tokens=800,
                        stop=STOP_DEFAULT
                    )
                    summary = tr or summary

                st.session_state.chat_messages.append({"role": "assistant", "content": summary})
                st.write(summary)

# --------------- CLASSIC MODE ---------------
with tab_classic:
    st.markdown("### Upload & Summarize")
    uploaded = st.file_uploader(
        "Upload a PDF", type=["pdf"], key="classic_uploader",
        help="Drag & drop or choose a PDF file (max ~100MB)"
    )
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if uploaded:
        raw = uploaded.read()
        with st.spinner("Extracting text from PDF…"):
            text = pdf_to_text(raw)

        st.markdown(
            f'<span class="badge">Extracted: {len(text):,} chars</span>'
            f'<span class="badge">Chunks: ~{max(1, (len(text)//max(1, MAX_CHARS))+1)}</span>',
            unsafe_allow_html=True,
        )

        if st.toggle("Show extracted text", False, key="classic_show"):
            st.text_area("Extracted Text", text, height=280)

        st.divider()
        c1, c2 = st.columns([1, 2], vertical_alignment="center")
        with c1:
            go = st.button("✨ Summarize (English, Long)", use_container_width=True, key="classic_go")
        with c2:
            st.markdown(
                "Tip: For fastest runs, keep PDFs under ~50 pages. "
                "Tweak token budgets in the sidebar if you see truncation."
            )

        if go:
            with st.spinner("Loading model and generating summary…"):
                llm = load_llm()
                summary = summarize_document(llm, text)

            st.markdown("### 📌 Summary")
            if summary:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write(summary)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Model returned empty text. Try increasing MAX_TOKENS_FINAL or reducing MAX_CHARS.")
    else:
        st.info("Drop a PDF to get started. The app will extract text, chunk it, and synthesize a long English summary.")

    st.markdown('</div>', unsafe_allow_html=True)
