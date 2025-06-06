# app.py  –  standalone Gradio UI for rahuljishu/mpie_iitj
import os, re, subprocess, json, tempfile, datetime, shutil
import gradio as gr
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

MODEL_REPO = "rahuljishu/mpie_iitj"
CACHE_DIR  = "./hf_cache"      # Render persists this across restarts

# One-time weight pull (10–60 s)
if not os.path.exists(CACHE_DIR):
    print("⬇️  First-time download of model repo…")
snapshot_download(repo_id=MODEL_REPO, cache_dir=CACHE_DIR,
                  local_dir=CACHE_DIR, local_dir_use_symlinks=False)

ANALYZE = os.path.join(CACHE_DIR, "analyze.py")    # adjust if needed

# ───────── helper: run agent, parse result minimally ─────────
def run_agent(path):
    proc = subprocess.run(
        ["python", ANALYZE, "--data", path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout

def parse(stdout):
    best = re.search(r"Best column:\s*(.*)", stdout).group(1).strip()
    reward_line = re.search(r"Reward break-down:\s*({.*})", stdout).group(1)
    reward = json.loads(reward_line.replace("'", '"'))
    rels = []
    m = re.search(r"Top relations:\s*([\s\S]*?)\n\n", stdout)
    for line in m.group(1).strip().splitlines():
        g = re.match(r"(.*?)→(.*?)\s*deg=(\d+)\s*R²=([\d.]+)", line)
        if g:
            src, dst, deg, r2 = g.groups()
            rels.append(dict(src=src.strip(), dst=dst.strip(),
                             deg=int(deg), r2=float(r2)))
    return best, reward, rels, stdout

def bar_chart(rels):
    fig, ax = plt.subplots(figsize=(5,2.5))
    ax.barh([f"{r['src']}→{r['dst']}" for r in rels],
            [r['r2'] for r in rels])
    ax.set_xlabel("R²"); ax.set_title("Top relations"); ax.invert_yaxis()
    fig.tight_layout(); return fig

def pdf_from_text(text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    can = canvas.Canvas(tmp.name, pagesize=letter)
    t = can.beginText(40, 750); t.setFont("Helvetica", 11)
    for line in text.splitlines(): t.textLine(line)
    can.drawText(t); can.showPage(); can.save(); return tmp.name

# ───────── Gradio callback ─────────
def analyze(file_path):
    try:
        out = run_agent(file_path); best, reward, rels, raw = parse(out)
    except Exception as e:
        return f"### ❌ Error\n```\n{e}\n```", None, None

    md = f"### Best column: `{best}`\n\n" + \
         "#### Reward\n" + "\n".join(
        f"- **{k}**: {v:.3f}" for k,v in reward.items()) + "\n"
    fig = bar_chart(rels)
    pdf = pdf_from_text(raw)
    return md, fig, pdf

# ───────── UI layout ─────────
CSS = """
body{background:#f5f5f7!important;
     font-family:-apple-system,BlinkMacSystemFont,Inter,'Segoe UI',sans-serif}
.gr-button{border-radius:8px!important;padding:8px 18px!important}
.gr-box{border-radius:12px!important;box-shadow:0 4px 14px rgba(0,0,0,.06)!important}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='font-weight:600'>Pattern Discovery Engine</h1>"
                "<p style='color:#666'>IIT-Jodhpur • RL-powered</p>")
    file_in = gr.File(label="Upload CSV/TXT", type="filepath")
    run_btn = gr.Button("Analyze", variant="primary")
    md_out, plot_out, pdf_out = gr.Markdown(), gr.Plot(), gr.File()
    run_btn.click(analyze, file_in, [md_out, plot_out, pdf_out])

if __name__ == "__main__":
    demo.launch()        # local URL; add share=True to get a public 72h link
