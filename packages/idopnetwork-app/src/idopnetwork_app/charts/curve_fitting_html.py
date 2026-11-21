"""HTML/SVG renderer for the Curve Fitting figure.

Port of the approved v8 prototype: authored in points so exports carry an exact
physical size, with tight (content-measured) canvas whitespace. Parameters map
one-to-one onto matplotlib concepts (figsize, GridSpec, sharey, MaxNLocator,
set_xlim/set_ylim).
"""
from __future__ import annotations

import json
from typing import Any, Iterable

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Curve Fitting · 出版级控制 v8</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#f4f6fa;color:#1c304b;
       font:13px/1.45 -apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .page{max-width:1100px;margin:0 auto;padding:14px 16px 18px}
  main{min-width:0}
  .chart-tools{display:flex;gap:8px;align-items:center;justify-content:flex-end;margin:0 0 8px;font-size:.74rem;color:#4a5f7d}
  .chart-tools select{border:1px solid #bacde3;border-radius:4px;padding:2px 4px;font-size:.74rem;color:#1c304b;background:#fff}
  .chart-tools button{border:1px solid #bacde3;border-radius:5px;background:#fff;color:#1c304b;padding:5px 9px;font-size:.74rem;cursor:pointer}
  .chart-tools button:hover{background:#f1f6fd}
  .figure{background:#fff;border:1px solid #d4dfed;border-radius:6px;padding:14px;overflow:auto}
  #stage{position:absolute;left:-99999px;top:0;visibility:hidden;pointer-events:none}
  .meta{margin-top:8px;color:#4a5f7d;font-size:.71rem}
  .note{margin-top:9px;padding:7px 9px;border-left:2px solid #245db4;background:#f1f6fd;
        color:#374e69;font-size:.71rem}
</style>
</head>
<body>
<div class="page">

  <main>
    <div id="stage" aria-hidden="true"></div>
    <div class="chart-tools">
      <span class="dpi">PNG dpi
        <select id="dpi"><option>300</option><option selected>600</option></select>
      </span>
      <button id="svg">__SVG_LABEL__</button>
      <button id="png">__PNG_LABEL__</button>
    </div>
    <div class="figure" id="figure"></div>
    <p class="meta" id="meta"></p>
  </main>
</div>
<script>
const DATA = __DATA__;
const S = __SETTINGS__;
const NS = "http://www.w3.org/2000/svg";
const PT_PER_MM = 72 / 25.4;
const el = (n, a = {}) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const $ = id => document.getElementById(id);

function niceTicks(lo, hi, count) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  let step = mag;
  for (const m of [1, 2, 2.5, 5, 10]) {
    step = m * mag;
    if ((hi - lo) / step <= count + 0.5) break;
  }
  const out = [];
  let v = Math.ceil(lo / step) * step;
  while (v <= hi + step * 1e-9) { out.push(+v.toFixed(10)); v += step; }
  return out;
}

function settings() { return S; }

const range = arr => arr.reduce((a, v) => [Math.min(a[0], v), Math.max(a[1], v)],
                                [Infinity, -Infinity]);

function buildSvg() {
  const s = settings();
  const mm = v => v * PT_PER_MM;
  const halfLabel = (s.fsize * 0.62 * 4.4) / 2 + 2;
  const gapX = mm(s.gapXmm), gapY = mm(s.gapYmm);
  const cellW = mm(s.cellWmm), cellH = mm(s.cellHmm);
  const labelRoom = s.fsize * 1.7 + 2;          // bottom-row tick labels
  const titleRoom = (s.letters ? s.fsize + 3 : 0);
  const tickLabelW = s.fsize * 0.62 * 4.2;      // e.g. "1.00"
  const m = {l: 22, r: 4, t: 8, b: 5};
  // Axis titles hang off the plot grid, so the canvas padding is derived from them.
  const padL = Math.max(26, m.l + tickLabelW + 4 + (s.ylabel ? s.yoff + s.fsize * 0.7 : 0));
  const padB = Math.max(8, m.b + labelRoom + (s.xlabel ? s.xoff + s.fsize * 1.1 : 0));
  const pad = {l: padL, r: Math.max(8, halfLabel), t: 6, b: padB};
  const W = pad.l + pad.r + s.cols * cellW + (s.cols - 1) * gapX;
  const H = pad.t + pad.b + s.rows * cellH + (s.rows - 1) * gapY + titleRoom;

  const cells = DATA.panels.slice(0, s.rows * s.cols);
  const lastRow = Math.max(0, Math.ceil(cells.length / s.cols) - 1);

  let xlo = Infinity, xhi = -Infinity, shared = [Infinity, -Infinity];
  const per = [];
  cells.forEach(p => {
    const [a, b] = range(p.curveX);
    xlo = Math.min(xlo, a); xhi = Math.max(xhi, b);
    const [c, d] = range(p.ys.concat(p.curveY));
    per.push([c, d]);
    shared = [Math.min(shared[0], c), Math.max(shared[1], d)];
  });
  const xSpan = xhi - xlo;
  const xMin = xlo - xSpan * s.mxl, xMax = xhi + xSpan * s.mxr;
  const xTicks = niceTicks(xMin, xMax, s.xnt);
  const sharedYs = [shared[0] - (shared[1] - shared[0]) * s.myb,
                    shared[1] + (shared[1] - shared[0]) * s.myt];

  const svg = el("svg", {width: `${W}pt`, height: `${H}pt`, viewBox: `0 0 ${W} ${H}`,
                          xmlns: NS, role: "img", "aria-label": "曲线拟合出版图"});
  const style = el("style", {});
  style.textContent = `text{font-family:${s.font};fill:#1c304b}`;
  svg.appendChild(style);

  cells.forEach((p, i) => {
    const row = Math.floor(i / s.cols), col = i % s.cols;
    const ox = pad.l + col * (cellW + gapX);
    const oy = pad.t + titleRoom + row * (cellH + gapY);
    const pw = cellW - m.l - m.r, ph = cellH - m.t - m.b;

    let ylo, yhi;
    if (s.sharey) [ylo, yhi] = sharedYs;
    else {
      const [a, b] = per[i], span = b - a;
      ylo = a - span * s.myb; yhi = b + span * s.myt;
    }
    const x = v => ox + m.l + (v - xMin) / (xMax - xMin) * pw;
    const y = v => oy + m.t + (1 - (v - ylo) / (yhi - ylo)) * ph;
    const yTicks = niceTicks(ylo, yhi, s.ynt);

    if (s.letters) {
      const letter = el("text", {x: ox, y: oy - 4, "font-size": s.fsize + 1,
                                 "font-weight": 700});
      letter.textContent = String.fromCharCode(97 + i);
      svg.appendChild(letter);
    }
    if (s.grid) {
      yTicks.forEach(v => svg.appendChild(el("line", {
        x1: ox + m.l, x2: ox + m.l + pw, y1: y(v), y2: y(v),
        stroke: "#e2e8f2", "stroke-width": 0.4})));
    }
    svg.appendChild(el("rect", {x: ox + m.l, y: oy + m.t, width: pw, height: ph,
                                fill: "none", stroke: "#22303f", "stroke-width": 0.5}));

    const labelW = s.fsize * 0.62 * 4.4;
    const spacing = xTicks.length > 1 ? pw / (xTicks.length - 1) : pw;
    const skip = Math.max(1, Math.ceil((labelW * 1.15) / spacing));
    xTicks.forEach((v, ti) => {
      const px = x(v);
      svg.appendChild(el("line", {x1: px, x2: px, y1: oy + m.t + ph, y2: oy + m.t + ph - 2.4,
                                  stroke: "#22303f", "stroke-width": 0.5}));
      if (row === lastRow && ti % skip === 0) {
        const t = el("text", {x: px, y: oy + m.t + ph + s.fsize * 1.35,
                              "text-anchor": "middle", "font-size": s.fsize});
        t.textContent = v.toFixed(2);
        svg.appendChild(t);
      }
    });
    const showY = !s.sharey || col === 0;
    const yDigits = Math.abs(yhi - ylo) < 2 ? 2 : 1;
    yTicks.forEach(v => {
      const py = y(v);
      svg.appendChild(el("line", {x1: ox + m.l, x2: ox + m.l + 2.4, y1: py, y2: py,
                                  stroke: "#22303f", "stroke-width": 0.5}));
      if (showY) {
        const t = el("text", {x: ox + m.l - 3, y: py + s.fsize * 0.34,
                              "text-anchor": "end", "font-size": s.fsize});
        t.textContent = v.toFixed(yDigits);
        svg.appendChild(t);
      }
    });

    const clipId = `c${i}`;
    const defs = el("defs", {});
    const clip = el("clipPath", {id: clipId});
    clip.appendChild(el("rect", {x: ox + m.l, y: oy + m.t, width: pw, height: ph}));
    defs.appendChild(clip);
    svg.appendChild(defs);
    const marks = el("g", {"clip-path": `url(#${clipId})`});
    const d = p.curveX.map((cx, k) =>
      `${k ? "L" : "M"}${x(cx).toFixed(2)},${y(p.curveY[k]).toFixed(2)}`).join(" ");
    marks.appendChild(el("path", {d, fill: "none", stroke: s.ccolor,
                                  "stroke-width": s.lw, "stroke-linecap": "butt"}));
    p.xs.forEach((cx, k) => marks.appendChild(el("circle", {cx: x(cx), cy: y(p.ys[k]),
      r: s.msize / 2, fill: "none", stroke: s.dcolor, "stroke-width": s.lw * 0.75})));
    svg.appendChild(marks);

    const cap = el("text", {x: ox + m.l + pw / 2, y: oy - 3, "text-anchor": "middle",
                            "font-size": s.fsize});
    cap.textContent = p.name;
    svg.appendChild(cap);
  });

  const gridLeft = pad.l + m.l;
  const gridRight = pad.l + (s.cols - 1) * (cellW + gapX) + m.l + (cellW - m.l - m.r);
  const gridTop = pad.t + titleRoom + m.t;
  const gridBottom = pad.t + titleRoom + lastRow * (cellH + gapY) + m.t + (cellH - m.t - m.b);
  if (s.xlabel) {
    const ty = gridBottom + labelRoom + s.xoff;
    const t = el("text", {x: (gridLeft + gridRight) / 2, y: ty, "text-anchor": "middle",
                          "font-size": s.fsize});
    t.textContent = s.xlabel;
    svg.appendChild(t);
  }
  if (s.ylabel) {
    const tx = gridLeft - tickLabelW - 4 - s.yoff;
    const cy = (gridTop + gridBottom) / 2;
    const t = el("text", {x: tx, y: cy, "text-anchor": "middle", "font-size": s.fsize,
                          transform: `rotate(-90 ${tx} ${cy})`});
    t.textContent = s.ylabel;
    svg.appendChild(t);
  }
  const halfLabelW = (s.fsize * 0.62 * 4.4) / 2;
  const ink = {
    left: gridLeft - tickLabelW - 4 - (s.ylabel ? s.yoff + s.fsize * 0.75 : 0) - 0.6,
    right: gridRight + halfLabelW + 0.6,
    top: pad.t - 0.6,
    bottom: gridBottom + labelRoom + (s.xlabel ? s.xoff + s.fsize * 1.2 : 0) + 0.6,
  };
  return {svg, W, H, cellW, cellH, shown: cells.length, ink};
}

function measureInk(svg) {
  const vb = (svg.getAttribute("viewBox") || "0 0 1 1").split(/\s+/).map(Number);
  const stage = $("stage");
  stage.appendChild(svg);
  const base = svg.getBoundingClientRect();
  const sx = base.width / vb[2], sy = base.height / vb[3];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  [...svg.querySelectorAll("text, rect, path, circle, line")].forEach(node => {
    if (node.closest("defs")) return;
    const b = node.getBoundingClientRect();
    if (!b.width && !b.height) return;
    minX = Math.min(minX, vb[0] + (b.left - base.left) / sx);
    minY = Math.min(minY, vb[1] + (b.top - base.top) / sy);
    maxX = Math.max(maxX, vb[0] + (b.right - base.left) / sx);
    maxY = Math.max(maxY, vb[1] + (b.bottom - base.top) / sy);
  });
  stage.removeChild(svg);
  if (!isFinite(minX) || maxX <= minX || maxY <= minY) return null;
  return {minX, minY, maxX, maxY};
}

function applyTight(svg, s, built) {
  if (!s.autocrop) return null;
  const ink = measureInk(svg);
  if (!ink) return null;
  const pad = s.edgePad + Math.max(s.lw, 0.5) / 2;
  const W = (ink.maxX - ink.minX) + pad * 2;
  const H = (ink.maxY - ink.minY) + pad * 2;
  svg.setAttribute("viewBox", `${ink.minX - pad} ${ink.minY - pad} ${W} ${H}`);
  svg.setAttribute("width", `${W}pt`);
  svg.setAttribute("height", `${H}pt`);
  return {W, H};
}

function render() {
  const s = settings();
  const built = buildSvg();
  const tight = applyTight(built.svg, s, built);
  const W = tight ? tight.W : built.W;
  const H = tight ? tight.H : built.H;
  $("figure").replaceChildren(built.svg);
  $("meta").textContent =
    `画布 ${(W / PT_PER_MM).toFixed(0)} × ${(H / PT_PER_MM).toFixed(0)} mm · ` +
    `子图 ${(built.cellW / PT_PER_MM).toFixed(0)} × ${(built.cellH / PT_PER_MM).toFixed(0)} mm · ` +
    `显示 ${built.shown} / ${DATA.panels.length} 个特征 · ${W.toFixed(0)} × ${H.toFixed(0)} pt · ` +
    (s.autocrop ? "自适应留白" : "固定留白");
}


function download(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("svg").onclick = () => {
  const s = settings();
  const built = buildSvg();
  const {svg, W} = built;
  const tight = applyTight(svg, s, built);
  const outMm = ((tight ? tight.W : W) / PT_PER_MM).toFixed(0);
  svg.setAttribute("xmlns", NS);
  download(`curve-fitting-${s.autocrop ? "tight-" : ""}${outMm}mm.svg`,
           new Blob([new XMLSerializer().serializeToString(svg)], {type: "image/svg+xml"}));
};
$("png").onclick = () => {
  const s = settings();
  const built = buildSvg();
  const {svg, W, H} = built;
  const tight = applyTight(svg, s, built);
  const outW = tight ? tight.W : W, outH = tight ? tight.H : H;
  svg.setAttribute("xmlns", NS);
  const text = new XMLSerializer().serializeToString(svg);
  const dpi = +$("dpi").value;
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(outW / 72 * dpi);
    canvas.height = Math.round(outH / 72 * dpi);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(b => download(`curve-fitting-${dpi}dpi.png`, b), "image/png");
  };
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(text);
};

render();
</script>
</body>
</html>
"""

DEFAULTS: dict[str, Any] = {
    "cell_w_mm": 41, "cell_h_mm": 25, "gap_x_mm": 0, "gap_y_mm": 5,
    "rows": 3, "cols": 4,
    "x_ticks": 3, "y_ticks": 2, "share_y": True,
    "margin_xl": 20, "margin_xr": 20, "margin_yb": 30, "margin_yt": 10,
    "font_family": "Times New Roman, Times, serif", "font_size": 11,
    "show_letters": True, "show_grid": False,
    "x_label": "拟动态时间", "y_label": "变换后的观测值",
    "x_title_offset": 10, "y_title_offset": 0,
    "marker_size": 6.5, "line_width": 1.5,
    "data_color": "#1f77b4", "curve_color": "#d62728",
    "edge_pad": 1.5, "autocrop": True, "dpi": 600,
    "svg_label": "导出 SVG（物理尺寸）", "png_label": "导出 PNG（按 dpi）",
}

_JS_KEYS = {
    "cell_w_mm": "cellWmm", "cell_h_mm": "cellHmm",
    "gap_x_mm": "gapXmm", "gap_y_mm": "gapYmm",
    "rows": "rows", "cols": "cols",
    "x_ticks": "xnt", "y_ticks": "ynt", "share_y": "sharey",
    "margin_xl": "mxl", "margin_xr": "mxr",
    "margin_yb": "myb", "margin_yt": "myt",
    "font_family": "font", "font_size": "fsize",
    "show_letters": "letters", "show_grid": "grid",
    "x_label": "xlabel", "y_label": "ylabel",
    "x_title_offset": "xoff", "y_title_offset": "yoff",
    "marker_size": "msize", "line_width": "lw",
    "data_color": "dcolor", "curve_color": "ccolor",
    "edge_pad": "edgePad", "autocrop": "autocrop",
    "dpi": "dpi",
}


def build_panels(quasi, samples, features: Iterable[str]) -> list[dict[str, Any]]:
    """Pack the two fitted tables into the renderer's panel payload."""
    x_all = [round(float(v), 4) for v in quasi.index]
    cx_all = [round(float(v), 4) for v in samples.index]
    panels = []
    for name in features:
        panels.append({
            "name": str(name),
            "xs": x_all,
            "ys": [round(float(v), 4) for v in quasi[name]],
            "curveX": cx_all,
            "curveY": [round(float(v), 4) for v in samples[name]],
        })
    return panels


def render(panels: list[dict[str, Any]], **settings: Any) -> str:
    """Return a self-contained HTML document with the figure and export buttons."""
    merged = {**DEFAULTS, **{k: v for k, v in settings.items() if v is not None}}
    # margins arrive as percentages; the renderer works in fractions
    for key in ("margin_xl", "margin_xr", "margin_yb", "margin_yt"):
        merged[key] = float(merged.get(key) or 0) / 100.0
    js_settings = {_JS_KEYS[key]: value for key, value in merged.items() if key in _JS_KEYS}
    html = TEMPLATE.replace("__DATA__", json.dumps({"panels": panels}, ensure_ascii=False))
    html = html.replace("__SETTINGS__", json.dumps(js_settings, ensure_ascii=False))
    html = html.replace("__SVG_LABEL__", str(merged["svg_label"]))
    html = html.replace("__PNG_LABEL__", str(merged["png_label"]))
    return html
