"""
Analyst report — HTML renderer.

One SELF-CONTAINED file: inline CSS + vanilla-JS canvas, zero external requests
(the production box is air-gapped). Design language inherited from the
clip06_timeline.html mock (CSS-variable theming, light/dark via
prefers-color-scheme + data-theme override).

Color roles (validated reference palette):
  traces     — one slate series color (single series per row → no legend box;
               the row header names it)
  flags      — amber WARNING bands/ticks, always paired with a count label
  direction  — diverging blue (suppressed / freeze) ↔ red (elevated / leakage)
               around a neutral midpoint; text stays in ink tokens, tint only
  percentile — sequential single-hue strip (alpha ramp of the series hue)
  ELAN       — green truth / red lie bands (validation mode only)
"""
from __future__ import annotations

import json
import math

_CSS = """
  :root {
    --paper:#f5f4f0; --ink:#23252a; --ink-soft:#6b6d72; --line:#dddbd4;
    --panel:#fbfaf7; --accent:#a86f1f; --trace:#5b7a94;
    --warn:#eda100; --pos:#e34948; --neg:#2a78d6;
    --lie:rgba(166,61,53,.13); --truth:rgba(63,122,82,.13);
    --lie-ink:#a63d35; --truth-ink:#3f7a52; --zero:#c9c6bd;
  }
  @media (prefers-color-scheme: dark) { :root {
    --paper:#181a1f; --ink:#e8e6e1; --ink-soft:#9a9ca1; --line:#2e3138;
    --panel:#1e2127; --accent:#d29a3d; --trace:#7d9cb8;
    --warn:#c98500; --pos:#e66767; --neg:#3987e5;
    --lie:rgba(196,84,74,.14); --truth:rgba(88,158,110,.14);
    --lie-ink:#d0766c; --truth-ink:#7dbb92; --zero:#3a3d45;
  }}
  :root[data-theme="dark"] {
    --paper:#181a1f; --ink:#e8e6e1; --ink-soft:#9a9ca1; --line:#2e3138;
    --panel:#1e2127; --accent:#d29a3d; --trace:#7d9cb8;
    --warn:#c98500; --pos:#e66767; --neg:#3987e5;
    --lie:rgba(196,84,74,.14); --truth:rgba(88,158,110,.14);
    --lie-ink:#d0766c; --truth-ink:#7dbb92; --zero:#3a3d45;
  }
  :root[data-theme="light"] {
    --paper:#f5f4f0; --ink:#23252a; --ink-soft:#6b6d72; --line:#dddbd4;
    --panel:#fbfaf7; --accent:#a86f1f; --trace:#5b7a94;
    --warn:#eda100; --pos:#e34948; --neg:#2a78d6;
    --lie:rgba(166,61,53,.13); --truth:rgba(63,122,82,.13);
    --lie-ink:#a63d35; --truth-ink:#3f7a52; --zero:#c9c6bd;
  }
  * { box-sizing: border-box; }
  body { background:var(--paper); color:var(--ink); margin:0;
    font-family:"Avenir Next","Segoe UI",system-ui,sans-serif; line-height:1.45; }
  .mono { font-family:"SF Mono","Cascadia Mono",Consolas,monospace; font-variant-numeric:tabular-nums; }
  main { max-width:1020px; margin:0 auto; padding:40px 24px 56px; }
  .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:11px; color:var(--accent); font-weight:600; }
  h1 { font-size:22px; font-weight:600; margin:2px 0 4px; }
  h2 { font-size:15px; font-weight:600; margin:34px 0 8px; border-top:1px solid var(--line); padding-top:18px; }
  h3 { font-size:13px; font-weight:600; margin:16px 0 6px; }
  .sub { color:var(--ink-soft); font-size:13.5px; max-width:78ch; margin:0 0 14px; }
  .chips { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 8px; }
  .chip { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:8px 12px; }
  .chip b { display:block; font-size:16px; font-weight:600; }
  .chip span { font-size:10.5px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.08em; }
  .note { font-size:12.5px; color:var(--ink-soft); }
  .alert { border-left:3px solid var(--warn); background:var(--panel); padding:8px 12px;
           font-size:13px; margin:8px 0; border-radius:0 4px 4px 0; }
  .alert.bad { border-left-color:var(--pos); }
  .alert.ok { border-left-color:var(--truth-ink); }
  table { border-collapse:collapse; font-size:12.5px; width:100%; margin:8px 0; }
  th { text-align:left; font-weight:600; color:var(--ink-soft); font-size:11px;
       text-transform:uppercase; letter-spacing:.06em; padding:6px 8px; border-bottom:1px solid var(--line); }
  td { padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
  .num { text-align:right; }
  .dir-up { color:var(--pos); font-weight:600; }
  .dir-dn { color:var(--neg); font-weight:600; }
  .scrollx { overflow-x:auto; }
  .row { display:grid; grid-template-columns:210px 1fr; gap:12px; align-items:center;
         border-top:1px solid var(--line); padding:8px 0; }
  .row h4 { font-size:12.5px; font-weight:600; margin:0; }
  .row .meta { font-size:11px; color:var(--ink-soft); }
  canvas { width:100%; height:64px; display:block; }
  canvas.strip { height:30px; }
  canvas.lane { height:16px; }
  .legend { display:flex; flex-wrap:wrap; gap:16px; font-size:12px; color:var(--ink-soft); margin:6px 0; }
  .sw { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-2px; margin-right:5px; }
  #tip { position:fixed; pointer-events:none; background:var(--panel); border:1px solid var(--line);
         border-radius:4px; padding:5px 9px; font-size:12px; display:none; z-index:10;
         box-shadow:0 2px 8px rgba(0,0,0,.12); }
  footer { margin-top:34px; border-top:1px solid var(--line); padding-top:14px;
           font-size:12px; color:var(--ink-soft); max-width:82ch; }
  @media (max-width:680px) { .row { grid-template-columns:1fr; gap:4px; } }
"""

_JS = """
const D = window.REPORT_DATA;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const mmss = ms => { const s = Math.round(ms/1000);
  return String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0'); };
const tip = document.getElementById('tip');

function prep(cv){ const dpr = window.devicePixelRatio||1, r = cv.getBoundingClientRect();
  cv.width = r.width*dpr; cv.height = r.height*dpr;
  const g = cv.getContext('2d'); g.scale(dpr,dpr); return [g, r.width, r.height]; }

function bands(g, clip, t0, t1, W, H){           // ELAN (validation) + flag bands
  const X = ms => (ms-t0)/(t1-t0)*W;
  if (clip.elan) for (const [lab,s,e] of clip.elan){
    const v = lab==='Lie' ? css('--lie') : lab==='Truth' ? css('--truth') : null;
    if (v){ g.fillStyle = v; g.fillRect(X(s),0,Math.max(1,X(e)-X(s)),H); } }
  g.fillStyle = css('--warn')+'2e';
  for (const f of clip.flags) g.fillRect(X(f.t),0,Math.max(2,X(f.t+2000)-X(f.t)),H);
}

function trace(cv, clip, vals, lim){             // one channel row
  const [g,W,H] = prep(cv);
  const t = clip.t, t0 = t[0], t1 = t[t.length-1]+2000;
  bands(g, clip, t0, t1, W, H);
  const L = lim||3, Y = z => H/2 - Math.max(-L,Math.min(L,z))/L*(H/2-4);
  g.strokeStyle = css('--zero'); g.lineWidth = 1;                       // zero + guides
  g.beginPath(); g.moveTo(0,Y(0)); g.lineTo(W,Y(0)); g.stroke();
  g.setLineDash([3,4]);
  for (const z of [2,-2]){ g.beginPath(); g.moveTo(0,Y(z)); g.lineTo(W,Y(z)); g.stroke(); }
  g.setLineDash([]);
  g.strokeStyle = css('--trace'); g.lineWidth = 2; g.lineJoin = 'round';
  let open = false; g.beginPath();
  for (let i=0;i<t.length;i++){
    const v = vals[i], x = (t[i]-t0)/(t1-t0)*W;
    if (v==null){ open=false; continue; }                              // null → gap
    if (!open){ g.moveTo(x,Y(v)); open=true; } else g.lineTo(x,Y(v));
  }
  g.stroke();
  cv.onmousemove = ev => { const r = cv.getBoundingClientRect();
    const ms = t0 + (ev.clientX-r.left)/r.width*(t1-t0);
    let k = 0; for (let i=0;i<t.length;i++) if (Math.abs(t[i]-ms)<Math.abs(t[k]-ms)) k=i;
    const v = vals[k];
    tip.style.display='block'; tip.style.left=(ev.clientX+14)+'px'; tip.style.top=(ev.clientY+10)+'px';
    tip.innerHTML = mmss(t[k]) + ' — <b>' + (v==null?'no data':v.toFixed(2)+'σ') + '</b>'; };
  cv.onmouseleave = () => tip.style.display='none';
}

function strip(cv, clip){                        // deviation-percentile strip
  const [g,W,H] = prep(cv);
  const t = clip.t, t0 = t[0], t1 = t[t.length-1]+2000;
  bands(g, clip, t0, t1, W, H);
  for (let i=0;i<t.length;i++){
    const p = clip.pct[i]; if (p==null) continue;
    const x0 = (t[i]-t0)/(t1-t0)*W, x1 = (t[i]+2000-t0)/(t1-t0)*W;
    g.fillStyle = css('--trace'); g.globalAlpha = 0.06 + 0.72*p;
    g.fillRect(x0, 3, Math.max(1,x1-x0-1), H-6); g.globalAlpha = 1;
  }
  g.strokeStyle = css('--warn'); g.lineWidth = 2;
  for (const f of clip.flags){ const x=(f.t-t0)/(t1-t0)*W;
    g.beginPath(); g.moveTo(x,0); g.lineTo(x,H); g.stroke(); }
  cv.onmousemove = ev => { const r = cv.getBoundingClientRect();
    const ms = t0 + (ev.clientX-r.left)/r.width*(t1-t0);
    let k = 0; for (let i=0;i<t.length;i++) if (Math.abs(t[i]-ms)<Math.abs(t[k]-ms)) k=i;
    tip.style.display='block'; tip.style.left=(ev.clientX+14)+'px'; tip.style.top=(ev.clientY+10)+'px';
    const p = clip.pct[k];
    tip.innerHTML = mmss(t[k]) + ' — deviation percentile <b>' + (p==null?'–':(100*p).toFixed(0)+'%') + '</b>'; };
  cv.onmouseleave = () => tip.style.display='none';
}

function lane(cv, clip, vals, name){             // coupling lane: sequential warm ramp
  const [g,W,H] = prep(cv);
  const t = clip.coupling.t, t0 = clip.t[0], t1 = clip.t[clip.t.length-1]+2000;
  for (let i=0;i<t.length;i++){
    const v = vals[i]; if (v==null) continue;
    const x0=(t[i]-t0)/(t1-t0)*W, x1=(t[i]+2000-t0)/(t1-t0)*W;
    g.fillStyle = css('--pos'); g.globalAlpha = Math.max(0, Math.min(1, v/5))*0.85;
    g.fillRect(x0,2,Math.max(1,x1-x0-1),H-4); g.globalAlpha = 1;
  }
  cv.onmousemove = ev => { const r = cv.getBoundingClientRect();
    const ms = t0 + (ev.clientX-r.left)/r.width*(t1-t0);
    let k = 0; for (let i=0;i<t.length;i++) if (Math.abs(t[i]-ms)<Math.abs(t[k]-ms)) k=i;
    const v = vals[k];
    tip.style.display='block'; tip.style.left=(ev.clientX+14)+'px'; tip.style.top=(ev.clientY+10)+'px';
    tip.innerHTML = name + ' ' + mmss(t[k]) + ' — <b>' + (v==null?'no data':v.toFixed(1)+'σ') + '</b>'; };
  cv.onmouseleave = () => tip.style.display='none';
}

for (const cv of document.querySelectorAll('canvas[data-kind]')){
  const clip = D.clips[+cv.dataset.clip], kind = cv.dataset.kind;
  if (kind==='strip') strip(cv, clip);
  else if (kind==='trace') trace(cv, clip, clip.traces[cv.dataset.ch]);
  else if (kind==='lane') lane(cv, clip, clip.coupling.nodes[cv.dataset.node], cv.dataset.node);
}
"""


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _clean(o):
    """Recursive NaN/inf → None at the JSON boundary. json.dumps would emit a bare
    NaN token (invalid JS) for any float that slipped the assembly guards; this is
    the single choke point that makes that structurally impossible."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _dir_cell(z):
    if z is None or not math.isfinite(z):
        return "<td class='num note'>–</td>"
    a = min(abs(z) / 3.0, 1.0) * 0.30
    var = "--pos" if z > 0 else "--neg"
    arrow = "▲" if z > 0 else "▼"
    return (f"<td class='num mono' style='background:"
            f"color-mix(in srgb, var({var}) {a*100:.0f}%, transparent)'>"
            f"{arrow} {z:+.2f}</td>")


def render(data) -> str:
    data = _clean(data)
    m, q = data["meta"], data["quality"]
    clips = data["clips"]
    h = []
    A = h.append

    A(f"<title>{_esc(m['recording_id'])} — Analyst Report</title>")
    A(f"<style>{_CSS}</style><main>")
    A(f"<span class='eyebrow'>Behavioral deviation attribution · {_esc(m['recording_id'])}"
      + (" · VALIDATION MODE (ground-truth overlay)" if m["validation_mode"] else "")
      + "</span>")
    A(f"<h1>Analyst report — {_esc(m['recording_id'])}</h1>")
    A("<p class='sub'>Every value is a z-score against <b>this subject's own neutral "
      "baseline</b> (clip 000). This report attributes <i>which behavioral channels "
      "deviated, when, and in which direction</i>. It renders <b>no verdict</b>: "
      "deviation is not deception, and interpretation belongs to the analyst.</p>")
    A("<div class='chips'>")
    for label, val in (("clips", m["n_clips"]), ("windows", m["n_windows"]),
                       ("features", m["n_features"]),
                       ("baseline windows", m["baseline"]["window_count"]),
                       ("flag rule", f"≥ p{int(m['flag_percentile']*100)}"),
                       ("generated", m["generated_on"] or "—")):
        A(f"<div class='chip'><b>{_esc(val)}</b><span>{_esc(label)}</span></div>")
    A("</div>")

    # ── data quality ─────────────────────────────────────────────────────────
    A("<h2 id='quality'>1 · Data quality & calibration health</h2>")
    bs = q["baseline_sanity"]
    if bs:
        if bs["degenerate"]:
            A(f"<div class='alert bad'><b>⚠ DEGENERATE BASELINE</b> — median deviation "
              f"{bs['median_dev']} vs expected ≈ {bs['expected_dev_sqrt_f']} (√F). A "
              f"near-zero baseline deviation is the failure signature (constant/dead "
              f"features), not health. Do not read attributions from this recording.</div>")
        else:
            A(f"<div class='alert ok'><b>✓ Baseline healthy</b> — baseline clip z "
              f"|mean| {bs['mean_abs_z']}, median deviation {bs['median_dev']} "
              f"(expected ≈ √F = {bs['expected_dev_sqrt_f']}).</div>")
    if q["dead_channels"]:
        A(f"<div class='alert bad'><b>⚠ {len(q['dead_channels'])} dead channel(s) "
          f"(100% missing)</b>: <span class='mono'>"
          + ", ".join(map(_esc, q["dead_channels"][:12]))
          + ("…" if len(q["dead_channels"]) > 12 else "") + "</span></div>")
    if m["baseline"]["uncalibratable"]:
        A(f"<div class='alert'><b>{len(m['baseline']['uncalibratable'])} channel(s) "
          f"constant during baseline</b> (z undefined, shown as gaps): <span class='mono'>"
          + ", ".join(map(_esc, m["baseline"]["uncalibratable"][:10])) + "</span></div>")
    if q["low_coverage"]:
        A("<div class='alert'><b>Low-coverage channels (&lt;50% of windows)</b>: "
          + ", ".join(f"<span class='mono'>{_esc(r['ch'])}</span> ({r['coverage']:.0%})"
                      for r in q["low_coverage"][:10]) + "</div>")
    A("<div class='scrollx'><table><tr><th>clip</th><th class='num'>windows</th>"
      "<th class='num'>mean confidence</th><th class='num'>target speaking</th>"
      "<th class='num'>flagged windows</th></tr>")
    for c in clips:
        role = " (baseline)" if c["is_baseline"] else ""
        aud = "–" if c["audio_active_frac"] is None else f"{c['audio_active_frac']:.0%}"
        conf = "–" if c["mean_confidence"] is None else f"{c['mean_confidence']:.2f}"
        A(f"<tr><td class='mono'>{c['file_index']:03d}{role}</td>"
          f"<td class='num mono'>{c['n_windows']}</td><td class='num mono'>{conf}</td>"
          f"<td class='num mono'>{aud}</td><td class='num mono'>{len(c['flags'])}</td></tr>")
    A("</table></div>")

    # ── overview strips ──────────────────────────────────────────────────────
    A("<h2 id='overview'>2 · Recording overview — deviation percentile per clip</h2>")
    A("<p class='note'>Darker = higher recording-wide deviation percentile. "
      f"<span class='sw' style='background:var(--warn)'></span>amber tick = flagged "
      f"window (≥ p{int(m['flag_percentile']*100)}).</p>")
    for i, c in enumerate(clips):
        A(f"<div class='row'><div><h4 class='mono'>clip {c['file_index']:03d}"
          + (" · baseline" if c["is_baseline"] else "")
          + f"</h4><div class='meta'>{len(c['flags'])} flagged</div></div>"
          f"<canvas class='strip' data-kind='strip' data-clip='{i}'></canvas></div>")

    # ── node table ───────────────────────────────────────────────────────────
    A("<h2 id='nodes'>3 · Direction-aware node summary (median z per channel family)</h2>")
    A("<p class='note'><span class='dir-dn'>▼ blue = suppressed vs baseline "
      "(freeze direction)</span> · <span class='dir-up'>▲ red = elevated "
      "(leakage direction)</span>. Text value is the family's median z over the clip.</p>")
    A("<div class='scrollx'><table><tr><th>clip</th>"
      + "".join(f"<th class='num'>{_esc(g)}</th>" for g in data["node_groups"]) + "</tr>")
    for row in data["node_table"]:
        A(f"<tr><td class='mono'>{row['file_index']:03d}</td>"
          + "".join(_dir_cell(cell["med_z"]) for cell in row["cells"]) + "</tr>")
    A("</table></div>")

    # ── per-clip traces ──────────────────────────────────────────────────────
    A("<h2 id='traces'>4 · Per-clip channel timelines (validated channels)</h2>")
    A("<p class='note'>Trace = channel z. Dotted guides at ±2σ; gaps = no usable "
      "data (never plotted as zero). "
      "<span class='sw' style='background:var(--warn);opacity:.4'></span>flagged window"
      + (" · <span class='sw' style='background:var(--truth)'></span>annotated Truth · "
         "<span class='sw' style='background:var(--lie)'></span>annotated Lie"
         if m["validation_mode"] else "") + "</p>")
    for i, c in enumerate(clips):
        if c["is_baseline"]:
            continue
        A(f"<h3 class='mono'>clip {c['file_index']:03d}</h3>")
        for ch, vals in c["traces"].items():
            cov = q["trace_coverage"].get(ch, 0)
            if all(v is None for v in vals):
                A(f"<div class='row'><div><h4 class='mono'>{_esc(ch)}</h4>"
                  f"<div class='meta'>no usable data in this recording</div></div>"
                  f"<div class='note'>—</div></div>")
                continue
            A(f"<div class='row'><div><h4 class='mono'>{_esc(ch)}</h4>"
              f"<div class='meta'>coverage {cov:.0%}</div></div>"
              f"<canvas data-kind='trace' data-clip='{i}' data-ch='{_esc(ch)}'></canvas></div>")

    # ── flagged windows ──────────────────────────────────────────────────────
    A("<h2 id='flags'>5 · Flagged windows — top contributing channels</h2>")
    nflags = sum(len(c["flags"]) for c in clips)
    if nflags == 0:
        A("<p class='note'>No window reached the flag threshold.</p>")
    else:
        A("<div class='scrollx'><table><tr><th>clip</th><th>time</th>"
          "<th class='num'>percentile</th><th>top contributing channels (signed z)</th></tr>")
        for c in clips:
            for f in c["flags"]:
                mm, ss = divmod(int(f["t"] // 1000), 60)
                tops = " · ".join(
                    f"<span class='mono'>{_esc(t['ch'])}</span> "
                    f"<span class='{'dir-up' if t['z'] > 0 else 'dir-dn'}'>"
                    f"{'▲' if t['z'] > 0 else '▼'}{t['z']:+.1f}</span>"
                    for t in f["top"])
                A(f"<tr><td class='mono'>{c['file_index']:03d}</td>"
                  f"<td class='mono'>{mm:02d}:{ss:02d}</td>"
                  f"<td class='num mono'>{f['pct']:.0%}</td><td>{tops}</td></tr>")
        A("</table></div>")

    # ── coupling lane (conditional) ──────────────────────────────────────────
    A("<h2 id='coupling'>6 · Cross-modal coupling</h2>")
    have_coupling = any(c["coupling"] for c in clips)
    if not have_coupling:
        A(f"<p class='note'>Not shown — {_esc(m['coupling_status'])}.</p>")
    else:
        A(f"<p class='note'>{_esc(m['coupling_status'])}. Lane intensity = how much "
          "less predictable the channel is from the subject's other channels vs "
          "baseline (decoupling), 0–5σ.</p>")
        for i, c in enumerate(clips):
            if not c["coupling"] or c["is_baseline"]:
                continue
            A(f"<h3 class='mono'>clip {c['file_index']:03d}</h3>")
            for node in c["coupling"]["nodes"]:
                A(f"<div class='row'><div><h4 class='mono'>{_esc(node)}</h4></div>"
                  f"<canvas class='lane' data-kind='lane' data-clip='{i}' "
                  f"data-node='{_esc(node)}'></canvas></div>")

    # ── footer ───────────────────────────────────────────────────────────────
    A("<footer><p><b>Provenance.</b> Source: <span class='mono'>"
      f"{_esc(m['source_csv'])}</span>; baseline fitted on "
      f"{m['baseline']['window_count']} windows of <span class='mono'>"
      f"{_esc(m['baseline']['source_csv'])}</span>.</p>"
      "<p><b>Doctrine.</b> This system performs per-subject baseline attribution, "
      "not classification. A flagged window means the subject deviated from their "
      "own neutral baseline — stress, cognitive load, topic salience and deception "
      "are all consistent with that observation. No channel here is a lie detector."
      "</p></footer>")
    A("<div id='tip'></div>")

    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    A(f"<script>window.REPORT_DATA={payload};</script>")
    A(f"<script>{_JS}</script></main>")
    return "\n".join(h)
