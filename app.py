# app.py  –  Gradio dashboard for rahuljishu/mpie_iitj (Render-ready)

import os, re, json, subprocess, tempfile, datetime
import gradio as gr
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ── 1. download model once ───────────────────────────────────────────
MODEL_REPO = "rahuljishu/mpie_iitj"
CACHE_DIR  = "/opt/render/project/hf_cache"         # persisted on Render

snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=CACHE_DIR,
    cache_dir=CACHE_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
)

ANALYZE = os.path.join(CACHE_DIR, "analyze.py")     # adjust if needed

# ── 2. helpers ───────────────────────────────────────────────────────
def run_agent(path):
    out = subprocess.check_output(["python", ANALYZE, "--data", path], text=True)
    return out

def parse(out):
    best = re.search(r"Best column:\s*(.*)", out).group(1).strip()
    reward = json.loads(
        re.search(r"Reward break-down:\s*({.*})", out).group(1).replace("'", '"')
    )
    rels = []
    for l in re.search(r"Top relations:\s*([\s\S]*?)\n\n", out).group(1).splitlines():
        m = re.match(r"(.*?)→(.*?)\s*deg=(\d+)\s*R²=([\d.]+)", l)
        if m:
            src, dst, deg, r2 = m.groups()
            rels.append(dict(src=src.strip(), dst=dst.strip(), deg=int(deg), r2=float(r2)))
    return best, reward, rels, out

def chart(rels):
    fig, ax = plt.subplots(figsize=(5,2.5))
    ax.barh([f"{r['src']}→{r['dst']}" for r in rels], [r['r2'] for r in rels])
    ax.invert_yaxis(); ax.set_xlabel("R²"); ax.set_title("Top relations")
    fig.tight_layout(); return fig

def pdf(text):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(f.name, pagesize=letter)
    t = c.beginText(40, 750); t.setFont("Helvetica", 11)
    for line in text.splitlines(): t.textLine(line)
    c.drawText(t); c.showPage(); c.save(); return f.name

# ── 3. Gradio callback ───────────────────────────────────────────────
def analyze(file_path):
    try:
        raw = run_agent(file_path)
        best, reward, rels, full = parse(raw)
    except Exception as e:
        return f"### ❌ Error\n```\n{e}\n```", None, None

    md = (
        f"### Best column: `{best}`\n\n"
        "#### Reward\n" + "\n".join(f"- **{k}**: {v:.3f}" for k,v in reward.items())
        + "\n\n#### Relations\n|Relation|Degree|R²|\n|---|---|---|\n"
        + "\n".join(f"|{r['src']}→{r['dst']}|{r['deg']}|{r['r2']:.3f}|" for r in rels)
        + f"\n\n*Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}*"
    )
    return md, chart(rels), pdf(full)

# ── 4. UI layout ─────────────────────────────────────────────────────
with gr.Blocks(css="body{background:#f5f5f7}") as demo:
    gr.Markdown("## Pattern Discovery Engine — IIT Jodhpur")
    inp = gr.File(label="Upload CSV/TXT", type="filepath")
    btn = gr.Button("Analyze", variant="primary")
    out_md, out_plot, out_pdf = gr.Markdown(), gr.Plot(), gr.File()
    btn.click(analyze, inp, [out_md, out_plot, out_pdf])

# ── 5. IMPORTANT: bind to 0.0.0.0 **and** $PORT ──────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
    )
