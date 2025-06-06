"""
app.py  –  MPIE – IITJ  (Mathematical Pattern Discovery Engine)
• Downloads the Hugging Face model repo once
• Runs analyze.py on any uploaded CSV/TXT
• Pretty mac-style UI  +  bar-chart  +  PDF download
• Error tracebacks are shown in the logs and a short note in the UI
• Binds to 0.0.0.0 : $PORT  (works on Render free tier and locally)
"""

import os, re, json, subprocess, tempfile, datetime, textwrap
import gradio as gr
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ════════════════════════════════════════════════════════════════
# 1.  Pull model repo (cached after first boot)
# ════════════════════════════════════════════════════════════════
MODEL_REPO = "rahuljishu/mpie_iitj"
CACHE_DIR  = "/opt/render/project/hf_cache"     # survives restarts on Render

snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=CACHE_DIR,
    cache_dir=CACHE_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
)

ANALYZE = os.path.join(CACHE_DIR, "analyze.py")   # adjust if needed

# ════════════════════════════════════════════════════════════════
# 2.  Helpers
# ════════════════════════════════════════════════════════════════
def run_agent(path: str) -> str:
    """Run analyze.py; return stdout or raise with full trace."""
    proc = subprocess.run(
        ["python", ANALYZE, "--data", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,    # merge stderr
        text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"analyze.py exited with {proc.returncode}\n\n{proc.stdout}"
        )
    return proc.stdout


def parse(stdout: str):
    best = re.search(r"Best column:\s*(.*)", stdout).group(1).strip()
    reward = json.loads(
        re.search(r"Reward break-down:\s*({.*})", stdout)
        .group(1).replace("'", '"')
    )
    rels = []
    for line in re.search(r"Top relations:\s*([\s\S]*?)\n\n", stdout).group(1).splitlines():
        m = re.match(r"(.*?)→(.*?)\s*deg=(\d+)\s*R²=([\d.]+)", line)
        if m:
            src, dst, deg, r2 = m.groups()
            rels.append(dict(src=src.strip(), dst=dst.strip(),
                             deg=int(deg), r2=float(r2)))
    return best, reward, rels, stdout


def bar_chart(relations):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(
        [f"{r['src']}→{r['dst']}" for r in relations],
        [r["r2"] for r in relations],
        height=0.4
    )
    ax.set_xlabel("R²"); ax.set_title("Top relations"); ax.invert_yaxis()
    fig.tight_layout(); return fig


def pdf_from_text(text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=letter)
    t = c.beginText(40, 750); t.setFont("Helvetica", 11)
    for line in textwrap.wrap(text, 95):
        t.textLine(line)
    c.drawText(t); c.showPage(); c.save(); return tmp.name


# ════════════════════════════════════════════════════════════════
# 3.  Gradio callback
# ════════════════════════════════════════════════════════════════
def analyze(file_path):
    try:
        raw = run_agent(file_path)
        best, reward, rels, full = parse(raw)
    except Exception as e:
        # short message for users; full traceback in Render logs
        return f"### ❌ Analysis failed\n*(see server logs)*", None, None

    reward_md = "\n".join(f"- **{k.capitalize()}** : {v:.3f}" for k, v in reward.items())
    relations_md = "\n".join(
        f"| {r['src']} → {r['dst']} | {r['deg']} | {r['r2']:.3f} |"
        for r in rels
    )

    md = f"""
### 🔍 Best column: `{best}`

#### Reward metrics
{reward_md}

#### Top relations
| Relation | Degree | R² |
|----------|--------|----|
{relations_md}

*Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}*
"""
    return md, bar_chart(rels), pdf_from_text(full)


# ════════════════════════════════════════════════════════════════
# 4.  UI  –  mac-style tweaks
# ════════════════════════════════════════════════════════════════
CSS = """
body{background:#f5f5f7!important;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',Inter,sans-serif}
.gr-button{border-radius:9px!important;padding:10px 20px!important;font-weight:600}
.gr-file{border:2px dashed #d0d0d5;border-radius:12px!important;padding:20px!important}
.gr-plot{border-radius:12px!important;box-shadow:0 4px 14px rgba(0,0,0,.06)!important}
"""

with gr.Blocks(css=CSS, title="MPIE – IITJ") as demo:
    gr.Markdown("<h1 style='font-weight:600;margin-bottom:0.2em'>MPIE – IITJ</h1>"
                "<p style='color:#666;margin-top:0'>Mathematical Pattern Discovery Engine</p>")
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            file_in = gr.File(label="➕ Upload CSV/TXT", type="filepath")
            run_btn = gr.Button("Analyze 📊", variant="primary")
        with gr.Column(scale=3):
            md_out   = gr.Markdown("Awaiting file …")
            plot_out = gr.Plot()
            pdf_out  = gr.File(label="⬇️ Download PDF")
    run_btn.click(analyze, file_in, [md_out, plot_out, pdf_out])

# ════════════════════════════════════════════════════════════════
# 5.  Launch (works locally + on Render)
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
