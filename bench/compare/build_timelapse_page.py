"""Build the side-by-side convergence page from timelapse_frames.json.

Generated, not hand-written, for the same reason the README tables are: the
page states ~200 numbers, and a hand-edited page drifts from the JSON the
moment either is touched. Re-run after render_timelapse.py.

The one number here that is easy to get wrong is the metric resolution. The
curve MUST be scored at the training resolution. Scoring at 400px while the
models trained at 800 made our own curve appear to plateau at 27.2 dB and even
fall 2.5 dB, because a denser model aliases more at reduced resolution -- the
same effect --antialias exists to fix. The two checkpoints that looked like a
regression go 29.57 -> 30.53 when scored at 800. render_timelapse.py now
defaults --metric-res to 800 and records it; this page prints it on its face so
a reader can see which protocol produced the curve.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "bench" / "results" / "timelapse_frames.json"
OUT = ROOT / "bench" / "results" / "timelapse.html"           # artifact body
OUT_STANDALONE = ROOT / "bench" / "results" / "timelapse_standalone.html"

# metal-gauss leads because it is the subject of the page and readers scan
# left-to-right; the competitors follow in finish order. With an equal wall-clock
# budget every lane ends within ~60s of the others, so ordering no longer encodes
# anything about the result and may as well aid reading.
LANES = [
    ("metal-gauss", "metal-gauss", "this repo -- Metal kernels, MCMC, 100k cap -- 15,000 it"),
    ("brush", "Brush", "WGPU, stock defaults -- 7,000 it"),
    ("msplat", "msplat", "Nerfstudio-style ADC, stock defaults -- 19,000 it"),
    ("spirula", "spirula-studio", "Vulkan/MoltenVK, synthetic preset -- 5,500 it"),
]
THRESHOLDS = [20, 24, 27, 30]


def main() -> None:
    data = json.loads(SRC.read_text())
    cfg = data["config"]
    frames = data["frames"]

    lanes = []
    for key, label, note in LANES:
        fr = sorted((f for f in frames if f["impl"] == key), key=lambda f: f["t"])
        if not fr:
            raise SystemExit(f"no frames for {key} -- rerun render_timelapse.py")
        keep = ("t", "psnr", "ssim", "n", "jpg")
        lanes.append({"key": key, "label": label, "note": note,
                      "frames": [{k: f[k] for k in keep} for f in fr]})

    # "First to N dB" uses the FIRST crossing, and requires the curve to still
    # be there afterwards is NOT checked -- msplat peaks at 24.27 then falls to
    # 23.48 as its ADC prunes, and hiding that would flatter it. First crossing
    # is the honest reading of "how long until you have N dB".
    firsts = {}
    for ln in lanes:
        firsts[ln["key"]] = {
            str(t): next((f["t"] for f in ln["frames"] if f["psnr"] >= t), None)
            for t in THRESHOLDS}

    payload = {"cfg": cfg, "lanes": lanes, "firsts": firsts,
               "thresholds": THRESHOLDS}
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    # Two outputs, because they have different hosts. The Artifact tool wraps
    # the body it is given in its own <head>, so OUT must NOT carry a doctype;
    # opening the same file from disk needs one, so OUT_STANDALONE does. Both
    # are gitignored -- they are derived from timelapse_frames.json in under a
    # second, and committing 1.1 MB of generated HTML would undo the history
    # slimming for no gain. The JSON is the measurement; this is a view of it.
    OUT.write_text(html)
    head, body = html.split('<div class="wrap">', 1)
    OUT_STANDALONE.write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        + head + '</head>\n<body>\n<div class="wrap">' + body
        + '\n</body>\n</html>\n')
    kb = OUT.stat().st_size / 1024
    print(f"  {len(frames)} frames, {len(lanes)} lanes  ({kb/1024:.1f} MB)")
    print(f"    artifact body -> {OUT.name}")
    print(f"    standalone    -> {OUT_STANDALONE.name}")
    for ln in lanes:
        last = ln["frames"][-1]
        print(f"    {ln['label']:<12} done {last['t']:6.1f}s  {last['psnr']:.2f} dB")


TEMPLATE = r"""<title>Splats Against the Clock</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#E9EBEE; --panel:#F7F8FA; --sunk:#DFE2E7;
  --line:#C7CCD4; --hair:#D8DCE2;
  --ink:#141920; --dim:#5B626D;
  --stage:#0F1216; --stage-ink:#E9E6DF; --stage-line:#242A32;
  --ours:#8A6A0F; --msplat:#26697A; --brush:#6A4F6E; --spirula:#3F6B4A;
  --ours-soft:#BDA14233; --msplat-soft:#26697A22; --brush-soft:#6A4F6E22;
  --spirula-soft:#3F6B4A22;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#111419; --panel:#181C22; --sunk:#0C0F13;
    --line:#2B313A; --hair:#232830;
    --ink:#E9E6DF; --dim:#89909B;
    --stage:#0B0E12; --stage-ink:#E9E6DF; --stage-line:#20262E;
    --ours:#D9B94E; --msplat:#7FBECB; --brush:#BC9FB8; --spirula:#8FB894;
    --ours-soft:#D9B94E22; --msplat-soft:#7FBECB1F; --brush-soft:#BC9FB81F;
    --spirula-soft:#8FB8941F;
  }
}
:root[data-theme="dark"]{
  --ground:#111419; --panel:#181C22; --sunk:#0C0F13;
  --line:#2B313A; --hair:#232830;
  --ink:#E9E6DF; --dim:#89909B;
  --stage:#0B0E12; --stage-ink:#E9E6DF; --stage-line:#20262E;
  --ours:#D9B94E; --msplat:#7FBECB; --brush:#BC9FB8; --spirula:#8FB894;
  --ours-soft:#D9B94E22; --msplat-soft:#7FBECB1F; --brush-soft:#BC9FB81F;
  --spirula-soft:#8FB8941F;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:Archivo,"Helvetica Neue",Arial,sans-serif;
  font-size:15px; line-height:1.5;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:34px 22px 60px;
      display:flex; flex-direction:column; gap:26px}

/* ---------- header ---------- */
.top{display:grid; grid-template-columns:1fr auto; gap:24px; align-items:end;
     border-bottom:2px solid var(--ink); padding-bottom:16px}
h1{margin:0; font-size:clamp(30px,5vw,50px); font-weight:800;
   font-stretch:82%; letter-spacing:-.015em; line-height:1; text-wrap:balance}
.sub{margin:9px 0 0; color:var(--dim); max-width:60ch; font-size:14.5px}
.clock{text-align:right; font-family:"IBM Plex Mono",ui-monospace,monospace;
       font-variant-numeric:tabular-nums; line-height:1}
.clock b{display:block; font-family:Archivo,sans-serif; font-stretch:70%;
         font-weight:800; font-size:clamp(38px,7vw,64px); letter-spacing:-.02em}
.clock span{display:block; font-size:10px; letter-spacing:.16em;
            text-transform:uppercase; color:var(--dim); margin-top:6px}

.proto{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px;
       color:var(--dim); display:flex; flex-wrap:wrap; gap:6px 18px;
       margin-top:-12px}
.proto b{color:var(--ink); font-weight:500}

/* ---------- lanes ---------- */
.stage{display:grid; grid-template-columns:repeat(4,1fr); gap:13px}
.lane{display:flex; flex-direction:column; gap:0; background:var(--panel);
      border:1px solid var(--line); border-radius:3px; overflow:hidden}
.lane[data-k="metal-gauss"]{border-color:var(--ours); box-shadow:0 0 0 1px var(--ours-soft)}
.rule{height:4px; background:var(--c)}
.lhead{padding:11px 13px 10px; display:flex; flex-direction:column; gap:2px;
       border-bottom:1px solid var(--hair)}
.lname{font-weight:700; font-size:16.5px; font-stretch:88%; letter-spacing:-.01em;
       color:var(--c)}
.lnote{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
       color:var(--dim); line-height:1.35}
.frame{position:relative; aspect-ratio:1/1; background:var(--stage);
       border-bottom:1px solid var(--hair)}
.frame img{position:absolute; inset:0; width:100%; height:100%; display:block;
           opacity:0; image-rendering:auto}
.frame img.on{opacity:1}
.wait{position:absolute; inset:0; display:grid; place-items:center;
      font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10.5px;
      letter-spacing:.14em; text-transform:uppercase; color:#5A616B}
.wait.off{display:none}
.stamp{position:absolute; right:9px; bottom:9px; transform:rotate(-7deg) scale(.86);
       opacity:0; transition:opacity .22s ease, transform .22s cubic-bezier(.2,1.5,.4,1);
       border:2px solid var(--c); border-radius:3px; padding:5px 9px 4px;
       background:#0F1216D9; text-align:center; pointer-events:none}
.stamp.on{opacity:1; transform:rotate(-7deg) scale(1)}
.stamp b{display:block; font-family:Archivo,sans-serif; font-stretch:70%;
         font-weight:800; font-size:19px; letter-spacing:.09em; color:var(--c);
         line-height:1}
.stamp span{display:block; font-family:"IBM Plex Mono",ui-monospace,monospace;
            font-size:9.5px; color:#B9BFC8; margin-top:3px;
            font-variant-numeric:tabular-nums}
.read{display:grid; grid-template-columns:repeat(3,1fr)}
.cell{padding:9px 4px 10px; text-align:center; border-right:1px solid var(--hair)}
.cell:last-child{border-right:0}
.cell u{display:block; text-decoration:none; font-family:"IBM Plex Mono",ui-monospace,monospace;
        font-size:9px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim)}
.cell v{display:block; font-family:"IBM Plex Mono",ui-monospace,monospace;
        font-variant-numeric:tabular-nums; font-size:16px; font-weight:600;
        margin-top:3px; color:var(--ink)}

/* ---------- transport ---------- */
.bar{display:flex; align-items:center; gap:14px; background:var(--panel);
     border:1px solid var(--line); border-radius:3px; padding:11px 14px}
button{font-family:inherit; color:var(--ink); background:var(--sunk);
       border:1px solid var(--line); border-radius:3px; cursor:pointer;
       padding:6px 12px; font-size:13px; font-weight:600}
button:hover{border-color:var(--ink)}
button:focus-visible{outline:2px solid var(--ours); outline-offset:2px}
#play{min-width:76px}
.speeds{display:flex; gap:5px; margin-left:auto}
.speeds button{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px;
               padding:5px 9px; font-weight:500}
.speeds button[aria-pressed="true"]{background:var(--ink); color:var(--ground);
                                    border-color:var(--ink)}
input[type=range]{flex:1; min-width:120px; accent-color:var(--ours); height:22px}
input[type=range]:focus-visible{outline:2px solid var(--ours); outline-offset:3px}

/* ---------- charts ---------- */
.charts{display:grid; grid-template-columns:1.55fr 1fr; gap:16px}
.chart{background:var(--panel); border:1px solid var(--line); border-radius:3px;
       padding:13px 14px 10px}
.ctitle{display:flex; justify-content:space-between; align-items:baseline;
        margin-bottom:8px}
.ctitle h2{margin:0; font-size:13px; font-weight:700; font-stretch:88%;
           letter-spacing:.02em}
.ctitle em{font-style:normal; font-family:"IBM Plex Mono",ui-monospace,monospace;
           font-size:10px; color:var(--dim)}
canvas{width:100%; height:190px; display:block}
.legend{display:flex; gap:15px; flex-wrap:wrap; margin-top:7px;
        padding-top:8px; border-top:1px solid var(--hair)}
.legend span{display:inline-flex; align-items:center; gap:6px;
             font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
             color:var(--dim)}
.legend i{width:15px; height:0; border-top:2.5px solid var(--c); display:inline-block}

/* ---------- footer table ---------- */
.tw{overflow-x:auto}
table{border-collapse:collapse; width:100%; font-size:13.5px}
caption{text-align:left; font-size:13px; font-weight:700; font-stretch:88%;
        padding-bottom:9px}
th,td{padding:8px 12px; border-bottom:1px solid var(--hair); text-align:right;
      font-family:"IBM Plex Mono",ui-monospace,monospace;
      font-variant-numeric:tabular-nums; white-space:nowrap}
th{font-size:10px; letter-spacing:.13em; text-transform:uppercase;
   color:var(--dim); font-weight:500; border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left; font-family:Archivo,sans-serif;
                              font-weight:600}
tr[data-k="metal-gauss"] td{background:var(--ours-soft)}
td.no{color:var(--dim)}
.note{font-size:13px; color:var(--dim); max-width:74ch}
.note b{color:var(--ink); font-weight:600}
.note a{color:inherit}

@media (max-width:1150px){
  .stage{grid-template-columns:repeat(2,1fr)}
}
@media (max-width:860px){
  .stage{grid-template-columns:1fr}
  .charts{grid-template-columns:1fr}
  .top{grid-template-columns:1fr; align-items:start}
  .clock{text-align:left}
}
@media (prefers-reduced-motion:reduce){
  .stamp{transition:none}
}
</style>

<div class="wrap">
  <header class="top">
    <div>
      <h1>Splats Against the Clock</h1>
      <p class="sub">Four Gaussian-splatting trainers reconstructing the same
      scene from the same seed point cloud. Every lane gets the <b>same
      ~390&nbsp;second budget</b> and runs however many iterations fit, because
      matching iteration counts across trainers whose per-step cost differs
      several-fold measures the schedule rather than the method. Seconds are
      what you actually wait.</p>
    </div>
    <div class="clock"><b id="t">0.0s</b><span>elapsed wall-clock</span></div>
  </header>

  <div class="proto" id="proto"></div>

  <section class="stage" id="stage"></section>

  <div class="bar">
    <button id="play" aria-label="Play or pause">Pause</button>
    <input type="range" id="scrub" min="0" max="1000" value="0" step="1"
           aria-label="Wall-clock position">
    <div class="speeds" id="speeds"></div>
  </div>

  <section class="charts">
    <div class="chart">
      <div class="ctitle"><h2>PSNR</h2><em>dB, higher is better</em></div>
      <canvas id="cp"></canvas>
      <div class="legend" id="lp"></div>
    </div>
    <div class="chart">
      <div class="ctitle"><h2>SSIM</h2><em>higher is better</em></div>
      <canvas id="cs"></canvas>
      <div class="legend" id="ls"></div>
    </div>
  </section>

  <div class="tw"><table id="ft"></table></div>

  <p class="note" id="foot"></p>
</div>

<script>
const D = __DATA__;
const CVAR = {"metal-gauss":"--ours", "msplat":"--msplat", "brush":"--brush",
              "spirula":"--spirula"};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const T_END = Math.max(...D.lanes.map(l => l.frames[l.frames.length-1].t));
const TAIL = 10;                       // hold the finished state before looping
const SPAN = T_END + TAIL;
let t = 0, playing = true, speed = 20, last = null;

/* ---------- build lanes ---------- */
const stage = document.getElementById("stage");
D.lanes.forEach(ln => {
  const fin = ln.frames[ln.frames.length-1];
  const el = document.createElement("article");
  el.className = "lane";
  el.dataset.k = ln.key;
  el.style.setProperty("--c", `var(${CVAR[ln.key]})`);
  el.innerHTML = `
    <div class="rule"></div>
    <div class="lhead">
      <div class="lname">${ln.label}</div>
      <div class="lnote">${ln.note}</div>
    </div>
    <div class="frame">
      <div class="wait">awaiting first checkpoint</div>
      <div class="stamp"><b>DONE</b><span>${fin.t.toFixed(0)}s &middot; ${fin.psnr.toFixed(2)} dB</span></div>
    </div>
    <div class="read">
      <div class="cell"><u>PSNR</u><v>&mdash;</v></div>
      <div class="cell"><u>SSIM</u><v>&mdash;</v></div>
      <div class="cell"><u>splats</u><v>&mdash;</v></div>
    </div>`;
  const frame = el.querySelector(".frame");
  ln.imgs = ln.frames.map((f, i) => {
    const im = new Image();
    im.src = "data:image/jpeg;base64," + f.jpg;
    im.alt = `${ln.label} at ${f.t.toFixed(1)} seconds`;
    frame.insertBefore(im, frame.firstChild);
    return im;
  });
  ln.el = el;
  ln.wait = el.querySelector(".wait");
  ln.stamp = el.querySelector(".stamp");
  ln.cells = [...el.querySelectorAll(".cell v")];
  ln.shown = -1;
  stage.appendChild(el);
});

/* ---------- protocol line, printed on the page's face ---------- */
const c = D.cfg;
document.getElementById("proto").innerHTML = [
  ["scene", c.scene],
  ["panel view", `${c.view_name} (held out)`],
  ["panel res", `${c.res}px`],
  ["metrics", `${c.metric_views} held-out views @ ${c.metric_res}px`],
  ["order", "strictly sequential, exclusive GPU"],
].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");

/* ---------- speeds ---------- */
const sp = document.getElementById("speeds");
[5, 20, 60].forEach(s => {
  const b = document.createElement("button");
  b.textContent = s + "×";
  b.setAttribute("aria-pressed", String(s === speed));
  b.onclick = () => {
    speed = s;
    [...sp.children].forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  };
  sp.appendChild(b);
});

/* ---------- first-to-quality table ---------- */
const ft = document.getElementById("ft");
ft.innerHTML = `<caption>Wall-clock to first reach a quality level</caption>
  <tr><th>implementation</th>${D.thresholds.map(x => `<th>${x} dB</th>`).join("")}
  <th>final</th></tr>` +
  D.lanes.map(ln => {
    const fin = ln.frames[ln.frames.length-1];
    const best = ln.frames.reduce((m, f) => Math.max(m, f.psnr), 0);
    return `<tr data-k="${ln.key}"><td>${ln.label}</td>` +
      D.thresholds.map(x => {
        const v = D.firsts[ln.key][String(x)];
        return v == null ? `<td class="no">never</td>` : `<td>${v.toFixed(1)}s</td>`;
      }).join("") +
      `<td>${fin.psnr.toFixed(2)} dB${best - fin.psnr > 0.05 ?
        ` <span class="no">(peak ${best.toFixed(2)})</span>` : ""}</td></tr>`;
  }).join("");

document.getElementById("foot").innerHTML =
  `Every implementation trains on the official train split from the same random point
   cloud and is scored by one evaluator on held-out views, so neither the holdout nor
   the metric differs between lanes. Competitors run their <b>shipped defaults</b>.
   Note that <b>msplat finishes first and finishes worst</b> &mdash; its adaptive
   density control prunes to ${(() => {
     const m = D.lanes.find(l => l.key === "msplat");
     return m.frames[m.frames.length-1].n.toLocaleString();
   })()}
   splats and its PSNR falls after peaking, so the DONE stamp carries each run's final
   quality: crossing the line early is not the same as winning.
   Runs were strictly sequential on one GPU; a contended run would invalidate every
   number on this page.
   <br><br>
   The step change in our curve near 147 s is the coarse-to-fine schedule reaching full
   resolution (<code>--num-downscales 2</code>): cheap quarter- and half-resolution steps end,
   per-step cost roughly doubles, and detail the 800&nbsp;px evaluation can finally see appears.
   PSNR and SSIM move together there and the splat count barely changes, so it is a
   resolution transition rather than densification.`;

/* ---------- legends: the traces are far from the coloured lane headers, so
   each chart names its own lines rather than relying on that association ---- */
["lp", "ls"].forEach(id => {
  document.getElementById(id).innerHTML = D.lanes.map(ln =>
    `<span style="--c:var(${CVAR[ln.key]})"><i></i>${ln.label}</span>`).join("");
});

/* ---------- charts ---------- */
function draw(canvas, key, lo, hi, fmt) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const x = canvas.getContext("2d");
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, w, h);
  const L = 40, R = 8, T = 8, B = 22;
  const px = v => L + (v / SPAN) * (w - L - R);
  const py = v => T + (1 - (v - lo) / (hi - lo)) * (h - T - B);
  const hair = css("--hair"), dim = css("--dim"), ink = css("--ink");

  x.font = '10px "IBM Plex Mono", monospace';
  x.strokeStyle = hair; x.lineWidth = 1;
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = lo + (hi - lo) * i / ticks, y = Math.round(py(v)) + .5;
    x.beginPath(); x.moveTo(L, y); x.lineTo(w - R, y); x.stroke();
    x.fillStyle = dim; x.textAlign = "right"; x.textBaseline = "middle";
    x.fillText(fmt(v), L - 6, y);
  }
  x.textAlign = "center"; x.textBaseline = "top";
  for (let s = 0; s <= SPAN; s += 60) {
    x.fillStyle = dim; x.fillText(s + "s", px(s), h - B + 6);
  }

  D.lanes.forEach(ln => {
    const col = css(CVAR[ln.key]);
    const pts = ln.frames.map(f => [px(f.t), py(f[key])]);
    // full curve as a ghost, traversed portion solid: the shape stays readable
    // while playback still shows how far each lane has actually got.
    x.lineWidth = 1.5; x.globalAlpha = .28; x.strokeStyle = col;
    x.beginPath(); pts.forEach((p, i) => i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]));
    x.stroke();
    x.globalAlpha = 1;
    const done = ln.frames.filter(f => f.t <= t);
    if (done.length) {
      x.lineWidth = ln.key === "metal-gauss" ? 2.6 : 1.9;
      x.beginPath();
      done.forEach((f, i) => {
        const p = [px(f.t), py(f[key])];
        i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]);
      });
      x.stroke();
      const lastF = done[done.length-1];
      x.fillStyle = col;
      x.beginPath(); x.arc(px(lastF.t), py(lastF[key]), 3.2, 0, 7); x.fill();
    }
  });

  x.strokeStyle = ink; x.globalAlpha = .45; x.lineWidth = 1;
  x.beginPath(); x.moveTo(Math.round(px(Math.min(t, SPAN))) + .5, T);
  x.lineTo(Math.round(px(Math.min(t, SPAN))) + .5, h - B); x.stroke();
  x.globalAlpha = 1;
}

const cp = document.getElementById("cp"), cs = document.getElementById("cs");
const allP = D.lanes.flatMap(l => l.frames.map(f => f.psnr));
const allS = D.lanes.flatMap(l => l.frames.map(f => f.ssim));
const P_LO = Math.floor(Math.min(...allP) - 1), P_HI = Math.ceil(Math.max(...allP) + 1);
const S_LO = Math.floor(Math.min(...allS) * 20) / 20, S_HI = 1.0;

function charts() {
  draw(cp, "psnr", P_LO, P_HI, v => v.toFixed(0));
  draw(cs, "ssim", S_LO, S_HI, v => v.toFixed(2));
}

/* ---------- per-tick render ---------- */
function render() {
  document.getElementById("t").textContent = Math.min(t, T_END).toFixed(1) + "s";
  document.getElementById("scrub").value = String(Math.round(t / SPAN * 1000));
  D.lanes.forEach(ln => {
    let i = -1;
    for (let k = 0; k < ln.frames.length; k++) if (ln.frames[k].t <= t) i = k;
    if (i !== ln.shown) {
      if (ln.shown >= 0) ln.imgs[ln.shown].classList.remove("on");
      if (i >= 0) {
        ln.imgs[i].classList.add("on");
        ln.wait.classList.add("off");
        const f = ln.frames[i];
        ln.cells[0].textContent = f.psnr.toFixed(2);
        ln.cells[1].textContent = f.ssim.toFixed(4);
        ln.cells[2].textContent = f.n.toLocaleString();
      } else {
        ln.wait.classList.remove("off");
        ln.cells.forEach(c => c.textContent = "—");
      }
      ln.shown = i;
    }
    ln.stamp.classList.toggle("on", t >= ln.frames[ln.frames.length-1].t);
  });
  charts();
}

function loop(now) {
  if (last === null) last = now;
  const dt = (now - last) / 1000; last = now;
  if (playing) {
    t += dt * speed;
    if (t >= SPAN) { t = 0; D.lanes.forEach(l => l.shown = -2); }
    render();
  }
  requestAnimationFrame(loop);
}

const playBtn = document.getElementById("play");
playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
};
document.getElementById("scrub").oninput = e => {
  t = Number(e.target.value) / 1000 * SPAN;
  playing = false; playBtn.textContent = "Play";
  render();
};
addEventListener("resize", charts);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", charts);
new MutationObserver(charts).observe(document.documentElement,
  {attributes: true, attributeFilter: ["data-theme"]});

if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
  playing = false; playBtn.textContent = "Play"; t = SPAN - TAIL;
}
render();
requestAnimationFrame(loop);
</script>
"""

if __name__ == "__main__":
    main()
