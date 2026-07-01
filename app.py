import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="VeritAI — Fake News Detector",
    page_icon="⚡",
    layout="centered"
)

# ── SAFE MODEL IMPORT ────────────────────────────────────────────
# If model.pkl / tfidf.pkl are missing or the classifier fails to load,
# show a clear error instead of letting Streamlit crash with a raw traceback.
try:
    from classifier import predict
    from explainer import explain, get_credibility_scores, SENSATIONAL_WORDS, CLICKBAIT_PHRASES
except FileNotFoundError as e:
    st.error(
        "⚠️ **Model files not found.**\n\n"
        f"{e}\n\n"
        "Run your training script first so `model/model.pkl` and "
        "`model/tfidf.pkl` exist, then reload this app."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Failed to load the model: {e}")
    st.stop()

# ── SAMPLE ARTICLES (for quick demo / testing) ───────────────────
SAMPLE_REAL = (
    "The State Bank of Pakistan announced on Monday that it will maintain "
    "its benchmark interest rate at 11% for the next quarter, citing stable "
    "inflation figures reported over the past three months. In a press "
    "briefing, the central bank's governor explained that the decision was "
    "based on a detailed review of consumer price data, import trends, and "
    "current account balances. Economists surveyed by local financial "
    "outlets largely expected this outcome, noting that a rate cut was "
    "unlikely given ongoing global commodity price pressures. The bank "
    "added that it would continue monitoring macroeconomic indicators and "
    "adjust policy as needed in future meetings."
)

SAMPLE_FAKE = (
    "SHOCKING: Scientists CONFIRM miracle fruit CURES everything overnight! "
    "You won't believe what doctors don't want you to know about this ONE "
    "WEIRD TRICK. Thousands are sharing this before it gets BANNED! Experts "
    "are baffled and the government is trying to hide the TRUTH from you. "
    "Share this NOW before it's deleted forever!!!"
)

# ── STYLES ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&family=Manrope:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #070709;
  --bg1:       #0e0e12;
  --bg2:       #141418;
  --bg3:       #1c1c22;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(255,255,255,0.12);
  --red:       #ff2d55;
  --red-glow:  rgba(255,45,85,0.18);
  --green:     #00e676;
  --green-glow:rgba(0,230,118,0.15);
  --amber:     #ffab00;
  --blue:      #2979ff;
  --purple:    #d500f9;
  --text:      #f0f0f5;
  --text2:     #8888a0;
  --text3:     #44445a;
  --mono:      'Space Mono', monospace;
  --sans:      'Manrope', sans-serif;
  --display:   'Syne', sans-serif;
}

html, body, [class*="css"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
}

.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.035'/%3E%3C/svg%3E");
  opacity: 0.4;
}

.ambient {
  position: fixed; top: -120px; left: 50%; transform: translateX(-50%);
  width: 700px; height: 400px; pointer-events: none; z-index: 0;
  background: radial-gradient(ellipse, rgba(255,45,85,0.07) 0%, transparent 70%);
}

.stApp { background: var(--bg) !important; }

.header-wrap { text-align: center; padding: 52px 0 36px; position: relative; }

.header-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 10px; letter-spacing: 3px;
  color: var(--red); text-transform: uppercase;
  border: 1px solid rgba(255,45,85,0.25);
  background: rgba(255,45,85,0.06);
  padding: 5px 14px; border-radius: 2px; margin-bottom: 24px;
}
.header-eyebrow::before {
  content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: var(--red); box-shadow: 0 0 8px var(--red);
  animation: pulse 2s ease infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } }

.header-title {
  font-family: var(--display); font-size: clamp(54px, 9vw, 96px);
  font-weight: 800; letter-spacing: -1px; line-height: 0.95;
  color: var(--text); margin-bottom: 6px;
}
.header-title .accent {
  color: transparent; -webkit-text-stroke: 1.5px var(--red); text-stroke: 1.5px var(--red);
}
.header-subtitle {
  font-family: var(--sans); font-size: 14px; font-weight: 300;
  color: var(--text2); letter-spacing: 0.3px; margin-top: 16px;
}

.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border2), transparent);
  margin: 8px 0 32px;
}

.input-card {
  background: var(--bg1); border: 1px solid var(--border);
  border-radius: 16px; padding: 28px; position: relative;
  overflow: hidden; margin-bottom: 20px;
}
.input-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,45,85,0.4), transparent);
}
.card-label {
  font-family: var(--mono); font-size: 10px; letter-spacing: 2.5px;
  color: var(--text3); text-transform: uppercase; margin-bottom: 14px;
}

textarea {
  background: var(--bg2) !important; border: 1px solid var(--border2) !important;
  border-radius: 10px !important; color: var(--text) !important;
  font-family: var(--sans) !important; font-size: 14px !important;
  line-height: 1.7 !important; transition: border-color 0.2s !important;
}
textarea:focus { border-color: rgba(255,45,85,0.5) !important; }
textarea::placeholder { color: var(--text3) !important; }

.stButton > button {
  width: 100% !important; background: var(--red) !important; color: #fff !important;
  border: none !important; border-radius: 10px !important; padding: 15px 28px !important;
  font-family: var(--display) !important; font-size: 18px !important; font-weight: 700 !important;
  letter-spacing: 3px !important; cursor: pointer !important; transition: all 0.2s !important;
  position: relative !important;
}
.stButton > button:hover {
  background: #ff1a40 !important;
  box-shadow: 0 0 40px rgba(255,45,85,0.5), 0 4px 20px rgba(255,45,85,0.3) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary (sample-article) buttons — smaller, outlined */
.sample-btn-row .stButton > button {
  background: var(--bg2) !important; color: var(--text2) !important;
  border: 1px solid var(--border2) !important; font-size: 12px !important;
  padding: 9px 14px !important; letter-spacing: 1px !important; font-family: var(--sans) !important;
  font-weight: 500 !important; box-shadow: none !important;
}
.sample-btn-row .stButton > button:hover {
  border-color: var(--red) !important; color: var(--text) !important;
  box-shadow: none !important; transform: none !important; background: var(--bg3) !important;
}

[data-testid="stFileUploader"] {
  background: var(--bg2) !important; border: 1.5px dashed var(--border2) !important;
  border-radius: 10px !important;
}

.char-counter {
  font-family: var(--mono); font-size: 10px; color: var(--text3);
  text-align: right; margin-top: 6px; letter-spacing: 1px;
}

.verdict-container {
  border-radius: 16px; padding: 32px; margin-bottom: 20px;
  position: relative; overflow: hidden; text-align: center;
}
.verdict-container.fake {
  background: linear-gradient(135deg, rgba(255,45,85,0.08) 0%, rgba(255,45,85,0.03) 100%);
  border: 1px solid rgba(255,45,85,0.3);
  box-shadow: 0 0 60px rgba(255,45,85,0.12), inset 0 1px 0 rgba(255,45,85,0.2);
}
.verdict-container.real {
  background: linear-gradient(135deg, rgba(0,230,118,0.08) 0%, rgba(0,230,118,0.02) 100%);
  border: 1px solid rgba(0,230,118,0.3);
  box-shadow: 0 0 60px rgba(0,230,118,0.1), inset 0 1px 0 rgba(0,230,118,0.2);
}
.verdict-container::before {
  content: ''; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(ellipse at 50% 0%, rgba(255,255,255,0.03), transparent 60%);
}
.verdict-overline {
  font-family: var(--mono); font-size: 10px; letter-spacing: 4px;
  color: var(--text3); text-transform: uppercase; margin-bottom: 12px;
}
.verdict-headline {
  font-family: var(--display); font-size: clamp(38px, 6vw, 56px);
  font-weight: 800; letter-spacing: 1px; line-height: 1; margin-bottom: 12px;
}
.verdict-headline.fake { color: var(--red); text-shadow: 0 0 40px rgba(255,45,85,0.5); }
.verdict-headline.real { color: var(--green); text-shadow: 0 0 40px rgba(0,230,118,0.4); }
.verdict-confidence { font-family: var(--mono); font-size: 13px; color: var(--text2); letter-spacing: 1px; }
.verdict-confidence span { font-size: 22px; font-weight: 700; }
.verdict-confidence span.fake { color: var(--red); }
.verdict-confidence span.real { color: var(--green); }

.sec-title {
  font-family: var(--mono); font-size: 10px; letter-spacing: 3px;
  color: var(--text3); text-transform: uppercase;
  display: flex; align-items: center; gap: 10px; margin: 28px 0 16px;
}
.sec-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border2), transparent); }

.spectrum-card { background: var(--bg1); border: 1px solid var(--border); border-radius: 14px; padding: 24px; margin-bottom: 16px; }
.spectrum-zone { font-family: var(--display); font-size: 18px; font-weight: 700; text-align: center; margin-bottom: 16px; letter-spacing: 1px; }
.spectrum-track {
  position: relative; height: 14px; border-radius: 7px;
  background: linear-gradient(90deg, var(--green) 0%, var(--amber) 50%, var(--red) 100%);
  margin-bottom: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.5);
}
.spectrum-needle {
  position: absolute; top: -5px; width: 6px; height: 24px; border-radius: 3px;
  background: #fff; transform: translateX(-50%);
  box-shadow: 0 0 12px rgba(255,255,255,0.9), 0 0 4px rgba(255,255,255,0.5);
  transition: left 1s cubic-bezier(0.4, 0, 0.2, 1);
}
.spectrum-labels {
  display: flex; justify-content: space-between; font-family: var(--mono);
  font-size: 9px; letter-spacing: 1px; color: var(--text3); text-transform: uppercase; margin-top: 8px;
}

.stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 4px; }
.stat-tile { background: var(--bg1); border: 1px solid var(--border); border-radius: 12px; padding: 18px 12px; text-align: center; transition: border-color 0.2s; }
.stat-tile:hover { border-color: var(--border2); }
.stat-val { font-family: var(--display); font-size: 30px; font-weight: 800; color: var(--text); line-height: 1; margin-bottom: 6px; }
.stat-key { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; color: var(--text3); text-transform: uppercase; }

.summary-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }
.summary-tile { background: var(--bg1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 10px; text-align: center; }
.summary-val { font-family: var(--display); font-size: 26px; font-weight: 800; line-height: 1; margin-bottom: 5px; }
.summary-key { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; color: var(--text3); text-transform: uppercase; }

.footer { text-align: center; padding: 32px 0 20px; font-family: var(--mono); font-size: 10px; letter-spacing: 2px; color: var(--text3); }

[data-testid="stDownloadButton"] > button {
  width: 100% !important; background: var(--bg2) !important; color: var(--text2) !important;
  border: 1px solid var(--border2) !important; border-radius: 10px !important; padding: 13px !important;
  font-family: var(--sans) !important; font-size: 14px !important; font-weight: 500 !important;
  transition: all 0.2s !important; letter-spacing: 0.5px !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: var(--bg3) !important; border-color: var(--red) !important; color: var(--text) !important;
}

.history-wrap { background: var(--bg1); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; margin-bottom: 16px; }
.history-header { padding: 16px 22px; border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: 10px; letter-spacing: 3px; color: var(--text3); text-transform: uppercase; }
.history-item { display: flex; align-items: center; gap: 16px; padding: 14px 22px; border-bottom: 1px solid var(--border); transition: background 0.15s; }
.history-item:last-child { border-bottom: none; }
.history-item:hover { background: var(--bg2); }
.hist-badge { font-family: var(--display); font-size: 14px; font-weight: 700; letter-spacing: 1px; flex-shrink: 0; min-width: 52px; }
.hist-badge.fake { color: var(--red); }
.hist-badge.real { color: var(--green); }
.hist-info { flex: 1; min-width: 0; }
.hist-preview { font-size: 12px; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }
.hist-meta { font-family: var(--mono); font-size: 10px; color: var(--text3); }
.hist-conf { font-family: var(--mono); font-size: 13px; font-weight: 700; flex-shrink: 0; }
.hist-conf.fake { color: var(--red); }
.hist-conf.real { color: var(--green); }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 760px !important; }
[data-testid="stDecoration"] { display: none; }
hr { border-color: var(--border) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg2) !important; border-radius: 8px !important; border: 1px solid var(--border) !important; padding: 3px !important; gap: 2px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 6px !important; color: var(--text2) !important; font-family: var(--sans) !important; font-size: 13px !important; }
.stTabs [aria-selected="true"] { background: var(--bg3) !important; color: var(--text) !important; }
.stSpinner > div { border-top-color: var(--red) !important; }
</style>

<div class="ambient"></div>
""", unsafe_allow_html=True)

# ── HELPER FUNCTIONS ─────────────────────────────────────────────

def render_metric_bar(name, score, color, icon):
    """Render one row of the Credibility Breakdown card."""
    bar_grad = f"linear-gradient(90deg,{color}99,{color})"
    return f"""
    <div style="margin-bottom:18px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:15px;">{icon}</span>
          <span style="font-family:'Manrope',sans-serif;font-size:13px;font-weight:500;color:#c0c0d0;">{name}</span>
        </div>
        <span style="font-family:'Space Mono',monospace;font-size:13px;font-weight:700;color:{color};">{score}%</span>
      </div>
      <div style="background:#1c1c22;border-radius:6px;height:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.06);">
        <div style="width:{score}%;height:100%;background:{bar_grad};border-radius:6px;box-shadow:0 0 10px {color}55;"></div>
      </div>
    </div>"""


def highlight_keywords(text, max_chars=1500):
    """Escape HTML then wrap flagged sensational/clickbait words in <mark>."""
    snippet = text[:max_chars]
    snippet = snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    found = []
    for w in sorted(SENSATIONAL_WORDS + CLICKBAIT_PHRASES, key=len, reverse=True):
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        if pattern.search(snippet):
            found.append(w)
            snippet = pattern.sub(
                f'<mark style="background:rgba(255,45,85,0.22);color:#ff6680;border-radius:4px;'
                f'padding:1px 5px;border-bottom:2px solid rgba(255,45,85,0.5);font-weight:600;">{w}</mark>',
                snippet)
    return snippet, found, len(text) > max_chars


# ── SESSION STATE ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "article_input" not in st.session_state:
    st.session_state.article_input = ""

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <div class="header-eyebrow">AI-Powered Analysis</div>
  <div class="header-title">VERIT<span class="accent">AI</span></div>
  <div class="header-subtitle">Paste any news article — we'll expose the truth in seconds.</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────
st.markdown('<div class="input-card"><div class="card-label">// Input Source</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📝  Paste Text", "📄  Upload File"])
article_text = ""

with tab1:
    # NOTE: sample-article buttons MUST be defined and handled BEFORE the
    # text_area widget below. Streamlit forbids writing to
    # st.session_state.article_input after the article_input widget has
    # already been instantiated in the same script run. By placing the
    # buttons first (and calling st.rerun() right after setting the state),
    # the assignment always happens either before the widget renders, or in
    # a fresh rerun where the widget hasn't been created yet.
    st.markdown('<div class="sample-btn-row">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🟢  Try Sample REAL", use_container_width=True):
            st.session_state.article_input = SAMPLE_REAL
            st.rerun()
    with c2:
        if st.button("🔴  Try Sample FAKE", use_container_width=True):
            st.session_state.article_input = SAMPLE_FAKE
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    inp = st.text_area(
        "", height=210, key="article_input",
        placeholder="Paste your news article here…\n\nTip: longer articles give more accurate results.",
        label_visibility="collapsed"
    )
    word_count_live = len(inp.split()) if inp else 0
    st.markdown(f'<div class="char-counter">{word_count_live} words</div>', unsafe_allow_html=True)

    if inp:
        article_text = inp

with tab2:
    uf = st.file_uploader("", type=["txt"], label_visibility="collapsed")
    if uf:
        try:
            article_text = uf.read().decode("utf-8")
            st.success(f"✅ {uf.name} — {len(article_text.split())} words loaded")
        except UnicodeDecodeError:
            st.error("⚠️ Couldn't read this file — please upload a plain UTF-8 .txt file.")
    # Only fall back to the pasted-text tab if no file was uploaded here,
    # so uploading a file doesn't silently override text you typed.
    elif inp:
        article_text = inp

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
go = st.button("⚡  ANALYZE NOW")

# ── ANALYSIS ──────────────────────────────────────────────────
if go:
    if len(article_text.strip()) < 20:
        st.warning("Please provide more text for accurate analysis (at least 20 characters).")
    else:
        try:
            with st.spinner("Analyzing…"):
                label, confidence = predict(article_text)
                reasons, polarity, subjectivity = explain(article_text, label)
                cred = get_credibility_scores(article_text, label, confidence)
        except Exception as e:
            st.error(f"⚠️ Something went wrong while analyzing this article: {e}")
            st.stop()

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "label": label, "confidence": confidence,
            "words": len(article_text.split()),
            "preview": article_text[:72] + "…",
        })

        st.markdown("<br>", unsafe_allow_html=True)

        # ── VERDICT ──
        cls = "fake" if label == "FAKE" else "real"
        icon = "🚨" if label == "FAKE" else "✅"
        verdict_text = "FAKE NEWS" if label == "FAKE" else "REAL NEWS"
        st.markdown(f"""
        <div class="verdict-container {cls}">
          <div class="verdict-overline">VERDICT</div>
          <div class="verdict-headline {cls}">{icon} {verdict_text}</div>
          <div class="verdict-confidence">Confidence &nbsp;<span class="{cls}">{confidence}%</span></div>
        </div>""", unsafe_allow_html=True)

        bar_col = "var(--red)" if label == "FAKE" else "var(--green)"
        st.markdown(f"""
        <div style="background:var(--bg3);border-radius:4px;height:5px;overflow:hidden;margin-bottom:28px;">
          <div style="width:{confidence}%;height:100%;background:{bar_col};border-radius:4px;
               box-shadow:0 0 12px {bar_col};transition:width 1.2s;"></div>
        </div>""", unsafe_allow_html=True)

        # ── SPECTRUM ──
        needle = int(confidence) if label == "FAKE" else int(100 - confidence)
        if needle < 30:
            zone, zc = "🟢 LIKELY REAL", "var(--green)"
        elif needle < 60:
            zone, zc = "🟡 UNCERTAIN", "var(--amber)"
        else:
            zone, zc = "🔴 LIKELY FAKE", "var(--red)"

        st.markdown(f"""
        <div class="sec-title">Credibility Spectrum</div>
        <div class="spectrum-card">
          <div class="spectrum-zone" style="color:{zc};">{zone}</div>
          <div class="spectrum-track"><div class="spectrum-needle" style="left:{needle}%;"></div></div>
          <div class="spectrum-labels"><span>Real</span><span>Uncertain</span><span>Fake</span></div>
        </div>""", unsafe_allow_html=True)

        # ── STATS ──
        words = len(article_text.split())
        chars = len(article_text)
        sents = len(re.findall(r'[.!?]+', article_text))
        st.markdown(f"""
        <div class="sec-title">Article Stats</div>
        <div class="stats-row">
          <div class="stat-tile"><div class="stat-val">{words}</div><div class="stat-key">Words</div></div>
          <div class="stat-tile"><div class="stat-val">{sents}</div><div class="stat-key">Sentences</div></div>
          <div class="stat-tile"><div class="stat-val">{chars}</div><div class="stat-key">Chars</div></div>
        </div>""", unsafe_allow_html=True)

        # ── CREDIBILITY BREAKDOWN ──
        avg_cred = int(sum(cred.values()) / len(cred))
        color_map = {"Language": "#d500f9", "Sentiment": "#ffab00", "Length": "#00e676", "Clickbait": "#ff2d55", "ML Model": "#2979ff"}
        icon_map = {"Language": "🔤", "Sentiment": "💬", "Length": "📏", "Clickbait": "🎯", "ML Model": "🤖"}
        rows_html = "".join(
            render_metric_bar(name, score, color_map.get(name, "#fff"), icon_map.get(name, "•"))
            for name, score in cred.items()
        )
        avg_col = "#00e676" if avg_cred >= 60 else ("#ffab00" if avg_cred >= 40 else "#ff2d55")
        st.markdown(f"""
        <div class="sec-title">Credibility Breakdown</div>
        <div style="background:#0e0e12;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;margin-bottom:16px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);"></div>
          {rows_html}
          <div style="border-top:1px solid rgba(255,255,255,0.07);padding-top:16px;margin-top:4px;display:flex;justify-content:space-between;align-items:center;">
            <span style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:2px;color:#44445a;text-transform:uppercase;">Overall Credibility</span>
            <span style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:{avg_col};text-shadow:0 0 20px {avg_col}66;">{avg_cred}%</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── SENTIMENT ──
        pol_pct = int((polarity + 1) / 2 * 100)
        sub_pct = int(subjectivity * 100)
        pol_col = "#ff2d55" if polarity < -0.2 else ("#00e676" if polarity > 0.2 else "#ffab00")
        sub_col = "#ff2d55" if subjectivity > 0.6 else ("#00e676" if subjectivity < 0.3 else "#ffab00")
        pol_label = "Negative 😟" if polarity < -0.2 else ("Positive 😊" if polarity > 0.2 else "Neutral 😐")
        sub_label = "Very Subjective" if subjectivity > 0.6 else ("Objective" if subjectivity < 0.3 else "Moderate")
        st.markdown(f"""
        <div class="sec-title">Sentiment Analysis</div>
        <div style="background:#0e0e12;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;margin-bottom:16px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);"></div>
          <div style="margin-bottom:20px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
              <span style="font-family:'Manrope',sans-serif;font-size:13px;font-weight:500;color:#c0c0d0;">😐 Polarity</span>
              <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-family:'Manrope',sans-serif;font-size:12px;color:{pol_col};">{pol_label}</span>
                <span style="font-family:'Space Mono',monospace;font-size:12px;color:#44445a;">{polarity:+.3f}</span>
              </div>
            </div>
            <div style="background:#1c1c22;border-radius:6px;height:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.06);">
              <div style="width:{pol_pct}%;height:100%;background:linear-gradient(90deg,{pol_col}88,{pol_col});border-radius:6px;box-shadow:0 0 10px {pol_col}55;"></div>
            </div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
              <span style="font-family:'Manrope',sans-serif;font-size:13px;font-weight:500;color:#c0c0d0;">🧠 Subjectivity</span>
              <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-family:'Manrope',sans-serif;font-size:12px;color:{sub_col};">{sub_label}</span>
                <span style="font-family:'Space Mono',monospace;font-size:12px;color:#44445a;">{subjectivity:.3f}</span>
              </div>
            </div>
            <div style="background:#1c1c22;border-radius:6px;height:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.06);">
              <div style="width:{sub_pct}%;height:100%;background:linear-gradient(90deg,{sub_col}88,{sub_col});border-radius:6px;box-shadow:0 0 10px {sub_col}55;"></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── KEYWORD HIGHLIGHTING ──
        highlighted, found_kws, truncated = highlight_keywords(article_text)
        kw_count = len(found_kws)
        if kw_count > 0:
            kw_badge = f'<span style="background:rgba(255,45,85,0.15);color:#ff2d55;border:1px solid rgba(255,45,85,0.3);border-radius:4px;padding:2px 8px;font-family:Space Mono,monospace;font-size:10px;letter-spacing:1px;">{kw_count} flagged</span>'
        else:
            kw_badge = '<span style="background:rgba(0,230,118,0.1);color:#00e676;border:1px solid rgba(0,230,118,0.3);border-radius:4px;padding:2px 8px;font-family:Space Mono,monospace;font-size:10px;">0 flagged</span>'
        st.markdown(f"""
        <div class="sec-title">Keyword Scan &nbsp; {kw_badge}</div>
        <div style="background:#0e0e12;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;margin-bottom:16px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,45,85,0.3),transparent);"></div>
          <div style="font-size:13px;line-height:2;color:#8888a0;font-family:'Manrope',sans-serif;font-weight:300;
               max-height:220px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#1c1c22 transparent;">
            {highlighted}{'…' if truncated else ''}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── REASONS ──
        reasons_html = ""
        for r in reasons:
            if r.startswith("✅"):
                dot_col, bg_col, bdr_col = "#00e676", "rgba(0,230,118,0.05)", "rgba(0,230,118,0.12)"
            elif "sensational" in r or "CAPS" in r or "short" in r or "Exaggerated" in r or "Clickbait" in r:
                dot_col, bg_col, bdr_col = "#ff2d55", "rgba(255,45,85,0.05)", "rgba(255,45,85,0.12)"
            else:
                dot_col, bg_col, bdr_col = "#ffab00", "rgba(255,171,0,0.05)", "rgba(255,171,0,0.12)"
            reasons_html += f"""
            <div style="display:flex;align-items:flex-start;gap:12px;padding:13px 16px;
                 background:{bg_col};border:1px solid {bdr_col};border-radius:10px;margin-bottom:8px;">
              <div style="width:8px;height:8px;border-radius:50%;background:{dot_col};
                   box-shadow:0 0 8px {dot_col};margin-top:4px;flex-shrink:0;"></div>
              <div style="font-size:13px;color:#a0a0b8;font-family:'Manrope',sans-serif;line-height:1.6;">{r}</div>
            </div>"""
        st.markdown(f"""
        <div class="sec-title">Why This Verdict?</div>
        <div style="margin-bottom:16px;">{reasons_html}</div>""", unsafe_allow_html=True)

        # ── DOWNLOAD + COPY ──
        report = f"""VERITAI — FAKE NEWS ANALYSIS REPORT
{'=' * 42}
Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Verdict    : {label}
Confidence : {confidence}%
Zone       : {zone}

ARTICLE STATS
-------------
Words      : {words}
Sentences  : {sents}
Characters : {chars}

SENTIMENT
---------
Polarity     : {polarity:+.3f}
Subjectivity : {subjectivity:.3f}

CREDIBILITY BREAKDOWN
---------------------
""" + "\n".join([f"{k:<12}: {v}%" for k, v in cred.items()]) + f"""
Overall    : {avg_cred}%

REASONS
-------
""" + "\n".join([f"• {r}" for r in reasons]) + f"""

ARTICLE PREVIEW
---------------
{article_text[:600]}...

{'=' * 42}
Powered by VeritAI — TF-IDF + Logistic Regression
"""
        st.markdown("<br>", unsafe_allow_html=True)
        dl_col, copy_col = st.columns(2)
        with dl_col:
            st.download_button(
                "📄  Download Report",
                data=report,
                file_name=f"veritai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain", use_container_width=True
            )
        with copy_col:
            # Simple clipboard-copy button using a tiny embedded HTML/JS component
            import streamlit.components.v1 as components
            escaped_report = report.replace("`", "\\`").replace("\\", "\\\\")
            components.html(f"""
                <button onclick="navigator.clipboard.writeText(`{escaped_report}`);
                    this.innerText='✅ Copied!'; setTimeout(()=>this.innerText='📋 Copy Report', 1500);"
                    style="width:100%;background:#141418;color:#8888a0;border:1px solid rgba(255,255,255,0.12);
                    border-radius:10px;padding:13px;font-family:'Manrope',sans-serif;font-size:14px;
                    font-weight:500;letter-spacing:0.5px;cursor:pointer;">📋 Copy Report</button>
            """, height=50)

        st.markdown('<div class="footer">VERITAI &nbsp;·&nbsp; TF-IDF + LOGISTIC REGRESSION &nbsp;·&nbsp; END SEM PROJECT</div>', unsafe_allow_html=True)

# ── HISTORY ───────────────────────────────────────────────────
if st.session_state.history:
    total = len(st.session_state.history)
    fakes = sum(1 for h in st.session_state.history if h["label"] == "FAKE")
    reals = total - fakes
    avg_c = int(sum(h["confidence"] for h in st.session_state.history) / total)

    st.markdown(f"""
    <div class="sec-title">Session History</div>
    <div class="summary-row">
      <div class="summary-tile"><div class="summary-val" style="color:var(--text);">{total}</div><div class="summary-key">Total</div></div>
      <div class="summary-tile"><div class="summary-val" style="color:var(--red);">{fakes}</div><div class="summary-key">Fake</div></div>
      <div class="summary-tile"><div class="summary-val" style="color:var(--green);">{reals}</div><div class="summary-key">Real</div></div>
      <div class="summary-tile"><div class="summary-val" style="color:var(--amber);">{avg_c}%</div><div class="summary-key">Avg Conf</div></div>
    </div>""", unsafe_allow_html=True)

    items_html = ""
    for h in reversed(st.session_state.history):
        cls = "fake" if h["label"] == "FAKE" else "real"
        icon = "🚨" if h["label"] == "FAKE" else "✅"
        items_html += f"""
        <div class="history-item">
          <div class="hist-badge {cls}">{icon} {h['label']}</div>
          <div class="hist-info">
            <div class="hist-preview">{h['preview']}</div>
            <div class="hist-meta">{h['time']} &nbsp;·&nbsp; {h['words']} words</div>
          </div>
          <div class="hist-conf {cls}">{h['confidence']}%</div>
        </div>"""

    st.markdown(f"""
    <div class="history-wrap">
      <div class="history-header">// Recent Analyses</div>
      {items_html}
    </div>""", unsafe_allow_html=True)

    if st.button("🗑️  Clear History"):
        st.session_state.history = []
        st.rerun()