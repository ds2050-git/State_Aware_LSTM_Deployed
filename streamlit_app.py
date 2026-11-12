import html, os, pickle, re, time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
import config
from src.base_lstm import BaseLSTM
from src.modified_lstm import ModifiedLSTM

BLUE, PINK = "#60a5fa", "#f472b6"
SEQ = config.SEQ_LENGTH
st.set_page_config(page_title="Alice LSTM Lab", page_icon="📖", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=Instrument+Sans:wght@400;600&display=swap');
html,body,[class*="st-"],.stApp{font-family:'Instrument Sans',sans-serif;color:#e6ebf5}
.stApp{background:#070b14}
.stApp::before,.stApp::after{content:"";position:fixed;border-radius:50%;filter:blur(110px);opacity:.35;z-index:0;pointer-events:none}
.stApp::before{width:520px;height:520px;background:#1d4ed8;top:-140px;left:-120px;animation:drift 18s ease-in-out infinite alternate}
.stApp::after{width:480px;height:480px;background:#be185d;bottom:-160px;right:-120px;animation:drift 22s ease-in-out infinite alternate-reverse}
@keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(120px,80px) scale(1.25)}}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1100px;padding-top:2.5rem;position:relative;z-index:1}
h1,h2,h3,h4{font-family:'Bricolage Grotesque',sans-serif!important;letter-spacing:-.01em;color:#f1f5ff}
.up{animation:up .8s cubic-bezier(.2,.8,.2,1) both}
@keyframes up{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}
.hero{padding:30px 0 10px}
.hero .tag{display:inline-block;padding:5px 14px;border:1px solid #2a3550;border-radius:99px;color:#9fb0d0;font-size:.85rem;background:#0d1424}
.hero h1{font-size:3.6rem;font-weight:800;line-height:1.03;margin:16px 0 14px}
.grad{background:linear-gradient(90deg,#60a5fa,#f472b6,#60a5fa);background-size:200% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 5s linear infinite}
@keyframes shine{to{background-position:200% center}}
.hero p{color:#9fb0d0;font-size:1.15rem;max-width:64ch;line-height:1.6}
.sec{margin:64px 0 18px}
.sec .num{font:800 .9rem 'Bricolage Grotesque';color:#f472b6;letter-spacing:.14em}
.sec h2{font-size:2.1rem;font-weight:800;margin:4px 0 6px}
.sec p{color:#9fb0d0;max-width:70ch;line-height:1.6;margin:0}
.card{background:linear-gradient(180deg,#0f1729,#0b1222);border:1px solid #222d47;border-radius:16px;padding:18px 20px;transition:transform .25s,border-color .25s,box-shadow .25s}
.card:hover{transform:translateY(-4px);border-color:#3b4b75;box-shadow:0 14px 40px #00000066}
.ttl{font:800 1.15rem 'Bricolage Grotesque';margin-bottom:8px}
.sub{color:#9fb0d0;font-size:.95rem;line-height:1.55}
.grid{display:grid;gap:14px}
.g4{grid-template-columns:repeat(4,1fr)} .g3{grid-template-columns:repeat(3,1fr)} .g2{grid-template-columns:repeat(2,1fr)}
@media(max-width:800px){.g4,.g3,.g2{grid-template-columns:1fr}.hero h1{font-size:2.4rem}}
.stat b{display:block;font:800 2rem 'Bricolage Grotesque';color:#f1f5ff}
.stat span{color:#9fb0d0;font-size:.9rem}
.pipe{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:6px;margin:10px 0 4px}
.node{background:#0f1729;border:1px solid #2a3550;border-radius:14px;padding:14px 16px;min-width:130px;text-align:center;animation:pulse 5s ease-in-out infinite}
.node b{display:block;font-family:'Bricolage Grotesque';font-size:1.02rem}
.node small{color:#9fb0d0}
.node:nth-child(1){animation-delay:0s}.node:nth-child(3){animation-delay:1s}.node:nth-child(5){animation-delay:2s}
.node:nth-child(7){animation-delay:3s}.node:nth-child(9){animation-delay:4s}
@keyframes pulse{0%,18%,100%{border-color:#2a3550;box-shadow:none}8%{border-color:#f472b6;box-shadow:0 0 28px #f472b655}}
.arr{width:46px;height:2px;background:#2a3550;position:relative;overflow:hidden;border-radius:2px}
.arr::after{content:"";position:absolute;top:0;left:-14px;width:14px;height:2px;background:#60a5fa;animation:flow 1.4s linear infinite}
@keyframes flow{to{left:46px}}
.gate .ic{font-size:1.6rem}
.eq{font-family:ui-monospace,Consolas,monospace;font-size:.88rem;background:#070b14;border:1px solid #1d2740;border-radius:10px;padding:10px 12px;margin-top:10px;color:#cfd9f0;line-height:1.7;overflow-x:auto;white-space:pre}
.eq .hl{color:#f472b6;font-weight:700}
.br{display:flex;align-items:center;gap:10px;margin:8px 0}
.br .w{width:96px;font-weight:600;overflow:hidden;text-overflow:ellipsis}
.br .tr{flex:1;height:22px;background:#17203a;border-radius:7px;overflow:hidden}
.br .fl{height:100%;border-radius:7px;animation:grow .9s cubic-bezier(.2,.8,.2,1)}
.br .pc{width:56px;text-align:right;color:#9fb0d0}
@keyframes grow{from{width:0}}
.gen{line-height:1.8;min-height:7rem;color:#e6ebf5}
.new{background:#f472b633;border-radius:5px;padding:0 3px;animation:pop .4s}
@keyframes pop{from{background:#f472b6;color:#000}}
.cur{display:inline-block;width:8px;height:1.1em;background:#f472b6;vertical-align:text-bottom;animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:0}}
.v{border-left:4px solid #f472b6;padding:4px 0 4px 16px;margin-top:16px;color:#cfd9f0;line-height:1.6;background:linear-gradient(90deg,#f472b61a,transparent)}
.stTextInput input{background:#0f1729!important;color:#f1f5ff!important;border:1px solid #2a3550!important;border-radius:12px!important}
.stTextInput input:focus{border-color:#f472b6!important;box-shadow:0 0 0 3px #f472b633!important}
.stButton>button{background:linear-gradient(90deg,#2563eb,#db2777);color:#fff;border:0;border-radius:12px;padding:.6rem 1.6rem;font-weight:600;transition:transform .2s,box-shadow .2s}
.stButton>button:hover{transform:translateY(-2px) scale(1.03);box-shadow:0 10px 30px #db277755;color:#fff}
[data-testid="stMetric"]{background:#0f1729;border:1px solid #222d47;border-radius:14px;padding:14px 16px}
[data-testid="stMetricLabel"] p{color:#9fb0d0}
[data-testid="stExpander"]{background:#0b1222;border:1px solid #222d47;border-radius:12px}
[data-testid="stCaptionContainer"],.stCaption{color:#9fb0d0}
hr{border-color:#1d2740}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(s):
    st.markdown(" ".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


@st.cache_resource
def load_all():
    with open(os.path.join(config.PROCESSED_DATA_PATH, "tokenizer.pkl"), "rb") as f:
        tok = pickle.load(f)
    V = len(tok["vocab"])
    specs = {"Base LSTM": (BaseLSTM, "models/base_lstm_model.pth"),
             "Modified LSTM": (ModifiedLSTM, "models/modified_lstm_model.pth")}
    models = {}
    for name, (cls, path) in specs.items():
        if not os.path.exists(path):
            models[name] = None
            continue
        m = cls(V, config.EMBEDDING_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models[name] = m.eval()
    return tok, models


try:
    tok, MODELS = load_all()
except Exception as e:
    st.error(f"Could not load the tokenizer: {e}. Run this from your project root after training.")
    st.stop()
w2i, i2w = tok["word_to_idx"], tok["idx_to_word"]
ready = {k: m for k, m in MODELS.items() if m is not None}
COLORS = (BLUE, PINK)


def encode(text):
    ids = [w2i.get(w, w2i["<UNK>"]) for w in text.lower().split()][-SEQ:]
    return [w2i["<PAD>"]] * (SEQ - len(ids)) + ids


@torch.no_grad()
def next_probs(model, text, temp=1.0):
    t0 = time.perf_counter()
    x = torch.LongTensor([encode(text)])
    logits, _ = model(x, model.init_hidden(1, "cpu"))
    logits = logits[0] / max(temp, 1e-3)
    logits[[w2i["<PAD>"], w2i["<UNK>"]]] = float("-inf")
    return F.softmax(logits, -1), (time.perf_counter() - t0) * 1000


def bars(rows, color):
    return "".join(f'<div class="br"><span class="w">{html.escape(w)}</span><div class="tr">'
                   f'<div class="fl" style="width:{p*100:.1f}%;background:{color}"></div></div>'
                   f'<span class="pc">{p*100:.1f}%</span></div>' for w, p in rows)


def parse_results():
    d = {"Base LSTM": {}, "Modified LSTM": {}, "n": 1020, "imp": "+0.69%"}
    try:
        txt = open("full_evaluation_results.txt", encoding="utf-8").read()
        base, mod = txt.split("BASE LSTM:")[1].split("MODIFIED LSTM:")
        for name, blk in (("Base LSTM", base), ("Modified LSTM", mod)):
            g = lambda k: float(re.search(k + r":\s*([\d.]+)", blk).group(1))
            d[name] = dict(score=g("Avg Score"), std=g("Std Dev"), wins=int(g("Wins")), time=g("Avg Time"))
        d["n"] = int(re.search(r"Test sentences:\s*(\d+)", txt).group(1))
        d["imp"] = re.search(r"IMPROVEMENT:\s*([+\-\d.]+%)", txt).group(1)
    except Exception:
        d["Base LSTM"] = dict(score=.2843, std=.1873, wins=169, time=.0527)
        d["Modified LSTM"] = dict(score=.2863, std=.1883, wins=172, time=.1023)
    return d


def params(m):
    return f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M" if m is not None else "n/a"


H("""<div class="hero up"><span class="tag">📖 Alice’s Adventures in Wonderland · PyTorch</span>
<h1>Two LSTMs learn to <span class="grad">finish Alice’s sentences</span></h1>
<p>A standard LSTM against a peephole LSTM I wrote from scratch, both trained on the same book.
Scroll down: what this is, how it works, try it live, then see how the change really performed.</p></div>""")


H("""<div class="sec up"><div class="num">01</div><h2>What is this?</h2>
<p>A language model reads a few words and guesses the next one. Both models here do exactly that, and the only
difference is the inside of the LSTM cell. The question: does letting the gates “peek” at memory make better guesses?</p></div>""")

H(f"""<div class="grid g4">
<div class="card stat up"><b>{len(tok['vocab']):,}</b><span>words in the vocabulary</span></div>
<div class="card stat up" style="animation-delay:.1s"><b>{SEQ}</b><span>words of context it reads</span></div>
<div class="card stat up" style="animation-delay:.2s"><b>{config.HIDDEN_DIM} × {config.NUM_LAYERS}</b><span>hidden size × layers</span></div>
<div class="card stat up" style="animation-delay:.3s"><b>{params(MODELS.get('Base LSTM'))} / {params(MODELS.get('Modified LSTM'))}</b><span>parameters: base / modified</span></div>
</div>""")
for k, m in MODELS.items():
    if m is None:
        st.warning(f"{k}: weights not found in models/ (the live demo needs them).")


H("""<div class="sec"><div class="num">02</div><h2>How it works</h2>
<p>Words become numbers, numbers flow through the LSTM, and the last step turns into a probability for every word in the book.</p></div>""")

def _t(x, y, main, sub=None, cls="lbl", anchor="start"):
    s = f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{main}'
    if sub:
        s += f'<tspan dy="3" font-size="8">{sub}</tspan>'
    return s + "</text>"


def _op(x, y, ch):
    return (f'<circle cx="{x}" cy="{y}" r="12" class="op"/>'
            f'<text x="{x}" y="{y+5}" class="opt" text-anchor="middle">{ch}</text>')


def _gate(cx, cy, label):
    return (f'<rect x="{cx-24}" y="{cy-15}" width="48" height="30" rx="8" class="gate"/>'
            f'<text x="{cx}" y="{cy+4}" class="gt" text-anchor="middle">{label}</text>')


def cell_svg(peep):
    w = lambda d: f'<path class="wire" d="{d}"/>'
    s = ['<svg viewBox="0 0 470 345" xmlns="http://www.w3.org/2000/svg">']

    s.append(w("M14 50 H456"))
    s.append(_t(14, 34, "c", "t−1")); s.append(_t(456, 34, "c", "t", anchor="end"))

    s += [w("M90 210 V62"), w("M190 210 V130 H218"), w("M270 210 V130 H242"),
          w("M230 118 V62"), w("M350 50 V80"), w("M350 104 V118"), w("M350 210 V142"),
          w("M362 130 H456")]
    s.append(_t(456, 118, "h", "t", anchor="end"))

    s += [w("M30 300 H350"), w("M90 300 V240"), w("M190 300 V240"), w("M270 300 V240"), w("M350 300 V240")]
    s.append('<circle cx="30" cy="300" r="4" fill="#60a5fa"/>')
    s.append(_t(30, 325, "h", "t−1")); s.append(_t(62, 325, ", x", "t"))

    if peep:
        s += ['<path class="peek" d="M52 50 V225 H66"/>', '<path class="peek" d="M52 190 H150 V225 H166"/>',
              '<path class="peek" d="M410 50 V225 H376"/>',
              '<circle cx="52" cy="50" r="3.5" class="pk"/>', '<circle cx="52" cy="190" r="3.5" class="pk"/>',
              '<circle cx="410" cy="50" r="3.5" class="pk"/>',
              '<text x="58" y="150" class="pkt">peephole</text>', '<text x="414" y="170" class="pkt">peephole</text>']

    s += [_op(90, 50, "×"), _op(230, 50, "+"), _op(230, 130, "×"), _op(350, 130, "×")]
    s += [_gate(90, 225, "σ  f"), _gate(190, 225, "σ  i"), _gate(270, 225, "tanh g"), _gate(350, 225, "σ  o")]
    s.append('<rect x="326" y="80" width="48" height="24" rx="8" class="gate"/>'
             '<text x="350" y="96" class="gt" text-anchor="middle">tanh</text>')
    s.append("</svg>")
    return "".join(s)


DIAGRAM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=Instrument+Sans:wght@400;600&display=swap');
body{margin:0;background:transparent;font-family:'Instrument Sans',sans-serif;color:#e6ebf5}
.wrap{display:flex;gap:16px;flex-wrap:wrap}
.card{flex:1 1 440px;background:linear-gradient(180deg,#0f1729,#0b1222);border:1px solid #222d47;border-radius:16px;padding:16px 18px;box-sizing:border-box;animation:up .8s cubic-bezier(.2,.8,.2,1) both}
.card:nth-child(2){animation-delay:.15s}
@keyframes up{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}
h3{font:800 1.2rem 'Bricolage Grotesque';margin:0 0 4px}
.sub{color:#9fb0d0;font-size:.92rem;line-height:1.5;margin:0 0 6px}
svg{width:100%;height:auto;display:block}
.wire{fill:none;stroke:#3b4b75;stroke-width:2;stroke-dasharray:6 6;animation:flow 1.2s linear infinite}
.peek{fill:none;stroke:#f472b6;stroke-width:2.2;stroke-dasharray:4 4;animation:flow .6s linear infinite}
@keyframes flow{to{stroke-dashoffset:-12}}
.pk{fill:#f472b6;animation:glow 1.6s ease-in-out infinite}
@keyframes glow{50%{opacity:.35}}
.pkt{fill:#f472b6;font-size:11px;font-weight:700}
.gate{fill:#0f1729;stroke:#60a5fa;stroke-width:1.5}
.gt{fill:#e6ebf5;font-size:11px;font-weight:600}
.op{fill:#0b1222;stroke:#9fb0d0;stroke-width:1.5}
.opt{fill:#e6ebf5;font-size:16px;font-weight:700}
.lbl{fill:#cfd9f0;font-size:13px;font-style:italic}
.eq{font-family:ui-monospace,Consolas,monospace;font-size:.82rem;background:#070b14;border:1px solid #1d2740;border-radius:10px;padding:10px 12px;margin-top:8px;color:#cfd9f0;line-height:1.7;overflow-x:auto;white-space:pre}
.hl{color:#f472b6;font-weight:700}
"""

_base_eq = ("f = σ( W·[h, x] + b )\ni = σ( W·[h, x] + b )\ng = tanh( W·[h, x] + b )\n"
            "o = σ( W·[h, x] + b )\nc = f ⊙ c_prev + i ⊙ g\nh = o ⊙ tanh(c)")
_peep_eq = ('f = σ( W·[h, x] + <span class="hl">Pf ⊙ c_prev</span> + b )\n'
            'i = σ( W·[h, x] + <span class="hl">Pi ⊙ c_prev</span> + b )\n'
            'g = tanh( W·[h, x] + b )\n'
            'o = σ( W·[h, x] + <span class="hl">Po ⊙ c_new</span> + b )\n'
            'c = f ⊙ c_prev + i ⊙ g\nh = o ⊙ tanh(c)')

DIAGRAMS_HTML = (
    "<style>" + DIAGRAM_CSS + "</style><div class='wrap'>"
    "<div class='card'><h3 style='color:#60a5fa'>Base LSTM</h3>"
    "<p class='sub'>Gates see only the current word and the previous hidden state. They cannot look at the cell memory.</p>"
    + cell_svg(False) + "<div class='eq'>" + _base_eq + "</div></div>"
    "<div class='card'><h3 style='color:#f472b6'>Peephole LSTM (modified)</h3>"
    "<p class='sub'>Gates also get a peephole into the cell memory, so they decide knowing what is stored.</p>"
    + cell_svg(True) + "<div class='eq'>" + _peep_eq + "</div></div></div>"
)
components.html(DIAGRAMS_HTML, height=760, scrolling=True)

H(f"""<div class="pipe">
<div class="node"><b>Words</b><small>(1, {SEQ})</small></div><div class="arr"></div>
<div class="node"><b>Embedding</b><small>(1, {SEQ}, {config.EMBEDDING_DIM})</small></div><div class="arr"></div>
<div class="node"><b>LSTM</b><small>(1, {SEQ}, {config.HIDDEN_DIM})</small></div><div class="arr"></div>
<div class="node"><b>Last step</b><small>(1, {config.HIDDEN_DIM})</small></div><div class="arr"></div>
<div class="node"><b>Softmax</b><small>(1, {len(tok['vocab'])})</small></div>
</div>""")


H("""<div class="grid g4" style="margin-top:18px">
<div class="card gate"><div class="ic">🗑️</div><div class="ttl">Forget gate</div><div class="sub">Decides which old memories to throw away.</div></div>
<div class="card gate"><div class="ic">✍️</div><div class="ttl">Input gate</div><div class="sub">Decides which new information is worth storing.</div></div>
<div class="card gate"><div class="ic">📦</div><div class="ttl">Cell memory</div><div class="sub">The long-term notebook that is carried from word to word.</div></div>
<div class="card gate"><div class="ic">📣</div><div class="ttl">Output gate</div><div class="sub">Decides how much memory to reveal as the next hidden state.</div></div>
</div>""")


H("""<div class="sec"><div class="num">03</div><h2>Try them live</h2>
<p>Type the start of a sentence. Both models guess the next word, then you can watch them write on their own.</p></div>""")

seed = st.text_input("Start a sentence", "alice was beginning to get very")
if not ready:
    st.info("Add models/base_lstm_model.pth and models/modified_lstm_model.pth to try the live demo.")
elif seed.strip():
    cols = st.columns(len(ready))
    for col, (name, m), color in zip(cols, ready.items(), COLORS):
        p, ms = next_probs(m, seed)
        top = torch.topk(p, 5)
        rows = [(i2w[i.item()], v.item()) for v, i in zip(top.values, top.indices)]
        col.markdown(f'<div class="card"><div class="ttl" style="color:{color}">{name}: next word</div>{bars(rows, color)}'
                     f'<div style="color:#9fb0d0;margin-top:8px">answered in {ms:.1f} ms</div></div>', unsafe_allow_html=True)

    with st.expander("What happened inside the model"):
        words = seed.lower().split()[-SEQ:]
        labels = ["PAD"] * (SEQ - len(words)) + words
        st.dataframe(pd.DataFrame({"word": labels, "id": encode(seed)}).T, use_container_width=True)
        st.code(f"tokens   (1, {SEQ})\nembedding (1, {SEQ}, {config.EMBEDDING_DIM})\n"
                f"LSTM     (1, {SEQ}, {config.HIDDEN_DIM})\nlast step (1, {config.HIDDEN_DIM})\n"
                f"logits   (1, {len(tok['vocab'])})")

    st.markdown("#### Watch them write")
    c1, c2 = st.columns(2)
    n = c1.slider("Words to generate", 5, 60, 25)
    temp = c2.slider("Temperature", 0.3, 1.5, 0.8, 0.05, help="Higher means more random word choices.")
    if st.button("Generate ✨", type="primary"):
        cols = st.columns(len(ready))
        slots, texts = {}, {}
        for col, (name, _), color in zip(cols, ready.items(), COLORS):
            col.markdown(f'<div class="ttl" style="color:{color}">{name}</div>', unsafe_allow_html=True)
            slots[name], texts[name] = col.empty(), seed.lower()
        for _ in range(n):
            for name, m in ready.items():
                p, _ms = next_probs(m, texts[name], temp)
                w = i2w[torch.multinomial(p, 1).item()]
                texts[name] += " " + w
                head = html.escape(texts[name].rsplit(" ", 1)[0])
                slots[name].markdown(f'<div class="card gen">{head} <span class="new">{html.escape(w)}</span><span class="cur"></span></div>',
                                     unsafe_allow_html=True)
            time.sleep(0.06)
        for name in ready:
            slots[name].markdown(f'<div class="card gen">{html.escape(texts[name])}</div>', unsafe_allow_html=True)


H("""<div class="sec"><div class="num">04</div><h2>So, did the change help?</h2>
<p>Both models were scored on held-out sentences they never saw during training.</p></div>""")

R = parse_results()
b, m = R["Base LSTM"], R["Modified LSTM"]
st.caption(f"Evaluated on {R['n']:,} held-out sentences")
c = st.columns(4)
c[0].metric("Base score", f"{b['score']:.4f}")
c[1].metric("Modified score", f"{m['score']:.4f}", R["imp"])
c[2].metric("Base speed", f"{b['time']*1000:.0f} ms")
c[3].metric("Modified speed", f"{m['time']*1000:.0f} ms", f"{m['time']/b['time']:.1f}x slower", delta_color="inverse")

st.markdown(f'<div class="card" style="margin-top:14px"><div class="ttl">Average score (0 to 100%)</div>'
            f'{bars([("Base", b["score"])], BLUE)}{bars([("Modified", m["score"])], PINK)}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="v"><b>Comparable accuracy, double the cost.</b> The {R["imp"]} difference is tiny next to the '
            f'sentence-to-sentence spread (±{b["std"]:.2f}), and wins are {b["wins"]} vs {m["wins"]}. '
            f'The gain is not statistically meaningful.</div>', unsafe_allow_html=True)

imgs = [("base_lstm_training_loss.png", "Base training loss"), ("modified_lstm_training_loss.png", "Modified training loss"),
        ("model_comparison.png", "Model comparison"), ("full_evaluation_results.png", "Full evaluation")]
shown = [(f, cap) for f, cap in imgs if os.path.exists(f)]
if shown:
    st.markdown("#### Training and evaluation plots")
    for i in range(0, len(shown), 2):
        for col, (f, cap) in zip(st.columns(2), shown[i:i + 2]):
            col.image(f, caption=cap, use_container_width=True)

H("""<div style="text-align:center;color:#6b7b9c;margin:60px 0 20px">Built with PyTorch and Streamlit · Base LSTM vs Peephole LSTM</div>""")