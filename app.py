"""GravityCheck demo page. Run: uv run app.py"""

import tempfile

import gradio as gr

from gravitycheck.analyze import Probe, analyze
from gravitycheck.render import render


def check(video, fall_prompt, size_lo_cm, size_hi_cm, walker_prompt):
    probes = []
    if fall_prompt.strip():
        probes.append(Probe(fall_prompt.strip(), "fall", (size_lo_cm / 100, size_hi_cm / 100)))
    if walker_prompt.strip():
        probes.append(Probe(walker_prompt.strip(), "gait"))
    a = analyze(video, probes)
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    render(a, out)
    v = a.verdict
    score = "--" if v.ai_score is None else f"{v.ai_score:.0f} / 100"
    lines = [f"## AI score: {score}", f"**{v.label.replace('_', ' ')}**", "", v.explanation, ""]
    for r in a.results:
        c = r.clock
        if c is None:
            lines.append(f"- `{r.probe.prompt}`: no usable clock")
        elif c.name == "gravity":
            lines.append(f"- `{r.probe.prompt}` falls at {c.detail['g_apparent_m_s2']:.2f} m/s² "
                         f"(Earth: 9.81) → clock {c.k:.2f}x [{c.lo:.2f}, {c.hi:.2f}]")
        else:
            lines.append(f"- `{r.probe.prompt}` walks at {c.detail['freq_hz']:.2f} steps/s "
                         f"(people: 1.6-2.0) → clock {c.k:.2f}x [{c.lo:.2f}, {c.hi:.2f}]")
    return out, "\n".join(lines)


with gr.Blocks(title="GravityCheck") as demo:
    gr.Markdown("# GravityCheck\nAI video can fool your eyes. It can't fool gravity.")
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Video")
            fall = gr.Textbox(label="Falling object", value="ball")
            with gr.Row():
                lo = gr.Number(label="Size from (cm)", value=6.0)
                hi = gr.Number(label="to (cm)", value=8.0)
            walker = gr.Textbox(label="Walking person", value="person")
            go = gr.Button("Check", variant="primary")
        with gr.Column():
            out_video = gr.Video(label="Evidence")
            report = gr.Markdown()
    go.click(check, [video, fall, lo, hi, walker], [out_video, report])

if __name__ == "__main__":
    demo.launch()
