"""
app.py  –  Stand-alone Gradio dashboard for rahuljishu/mpie_iitj
● Downloads model repo once with snapshot_download
● Presents polished mac-style UI
● Works locally (port 7860) and on Render (binds to $PORT)
"""

import os, re, json, subprocess, tempfile, datetime
import gradio as gr
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ────────────────────────────────────────────────────────
# 1. grab model weights the first time the container starts
# ────────────────────────────────────────────────────────
MODEL_REPO = "rahuljishu/mpie_iitj"
CACHE_DIR  = "./hf_cache"          # persists between Render restarts

if not os.path.exists(CACHE_DIR):
    print("⬇️  First-time download of model repo …", flush=True)
snapshot_download(
    repo_id=MODEL_REPO,
    cache_dir=CACHE_DIR,
    local_dir=CACHE_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
)

ANALYZE = os.path.join(CACHE_DIR, "analyze.py")  # adjust if path differs

# ────────────────────────────────────────────────────────
# 2. helpers – run agent, parse stdout, make chart & PDF
# ────────────────────────────────────────────────────────
def run_agent(data_path: str) -> str:
    proc = subprocess.run(
        ["python", ANALYZE, "--data", data_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


def parse(stdout: str):
    best = re.search(r"Best column:\s*(.*)", stdout).group(1).strip()
    reward = json.loads(
        re.search(r"Reward break-down:\s*({.*})", stdout)
        .group(1)
        .replace("'", '"')
    )
    relations = []
    block = re.search(r"Top relations:\s*([\s\S]*?)\n\n", stdout).group(1)
    for line in block.strip().splitlines():
        m = re.match(r"(.*?)→(.*?)\s*deg=(\d+)\s*R²=([\d.]+)", line)
        if m:
            src, dst, deg, r2 = m.groups()
            relations.append(
                {"src": src.strip(), "dst": dst.strip(), "deg": int(deg), "r2": float(r2)}
            )
    return best, reward, relations, stdout


def bar_chart(relations):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(
        [f"{r['src']}→{r['dst']}" for r in relations],
        [r["r2"] for r in relations],
    )
    ax.set_xlabel("R²")
    ax.set_title("Top relations")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def pdf_from_text(text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=letter)
    t = c.beginText(40, 750)
    t.setFont("Helvetica", 11)
    for line in text.splitlines():
        t.textLine(line)
    c.drawText(t)
    c.showPage()
    c.save()
    return tmp.name


# ────────────────────────────────────────────────────────
# 3. Gradio callback
# ────────────────────────────────────────────────────────
def analyze(file_path: str):
    try:
        raw = run_agent(file_path)
        best, reward, rels, full = parse(raw)
    except Exception as e:
        return f"### ❌ Error\n```\n{e}\n```", None, None

    reward_md = "\n".join(f"- **{k}**: {v:.3f}" for k, v in reward.items())
    table_md = "\n".join(
        f"| {r['src']} → {r['dst']} | {r['deg']} | {r['r2']:.3f} |" for r in rels
    )

    md = f"""
### 🔍 Best column: `{best}`

#### Reward
{reward_md}

#### Relations
| Relation | Degree | R² |
|----------|--------|----|
{table_md}

*Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}*
"""

    return md, bar_chart(rels), pdf_from_text(full)


# ────────────────────────────────────────────────────────
# 4. UI layout + CSS
# ────────────────────────────────────────────────────────
CSS = """
body{background:#f5f5f7!important;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',Inter,sans-serif}
.gr-button{border-radius:8px!important;padding:8px 18px!important}
.gr-box{border-radius:12px!important;box-shadow:0 4px 14px rgba(0,0,0,.06)!important}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "<h1 style='font-weight:600'>Pattern Discovery Engine</h1>"
        "<p style='color:#666'>IIT-Jodhpur • RL-powered</p>"
    )
    file_in = gr.File(label="Upload CSV/TXT", type="filepath")
    run_btn = gr.Button("Analyze", variant="primary")
    md_out, plot_out, pdf_out = gr.Markdown(), gr.Plot(), gr.File()
    run_btn.click(analyze, file_in, [md_out, plot_out, pdf_out])

# ────────────────────────────────────────────────
