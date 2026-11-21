"""HTML/SVG renderer for the Curve Fitting figure, with its own publication panel.

The collapsible panel (canvas/subplot sizes, ticks, shared Y, four-sided margins,
typography, axis titles, marker/line styling, SVG/PNG export) lives inside the
component, so the figure stays in sync without a server round-trip.
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
  .page{display:grid;grid-template-columns:268px minmax(0,1fr);gap:18px;max-width:1240px;margin:0 auto;padding:10px 12px}
  aside{background:#fff;border:1px solid #d4dfed;border-radius:8px;padding:10px 12px;height:max-content}
  aside > h2{margin:2px 0 6px;font:600 .78rem/1.3 Georgia,"Songti SC",serif;color:#193454;
              text-transform:uppercase;letter-spacing:.04em}
  details{border-top:1px solid #e6eef7;padding:2px 0}
  details:first-of-type{border-top:0}
  summary{cursor:pointer;list-style:none;padding:7px 0 5px;font-size:.72rem;font-weight:600;
          color:#4a5f7d;text-transform:uppercase;letter-spacing:.04em;display:flex;
          justify-content:space-between;align-items:center}
  summary::-webkit-details-marker{display:none}
  summary::after{content:"+";color:#9dbde0;font-size:.9rem;line-height:1}
  details[open] > summary::after{content:"−"}
  .body{padding-bottom:6px}
  .row{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:4px 0}
  .row span{color:#374e69;font-size:.73rem}
  .row input[type=number]{width:56px}
  .row input[type=text]{width:108px}
  .pair{display:flex;gap:6px;align-items:center}
  input,select{border:1px solid #bacde3;border-radius:4px;color:#1c304b;font-size:.73rem;
               padding:2px 4px;background:#fff}
  input[type=color]{width:30px;height:20px;padding:0}
  input[type=checkbox]{accent-color:#245db4}
  .btn{display:block;width:100%;margin-top:7px;padding:7px 9px;border:1px solid #bacde3;
       border-radius:5px;background:#fff;color:#1c304b;font-size:.75rem;cursor:pointer}
  .btn:hover{background:#ecf3fc}
  .btn.primary{background:#245db4;border-color:#245db4;color:#fff}
  .btn.primary:hover{background:#1d4d95}
  main{min-width:0}
  .legend{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;margin:0 0 8px;font-size:.74rem;color:#374e69}
  .legend .item{display:flex;align-items:center;gap:6px}
  .legend .swatch{display:inline-block;width:22px;height:0;border-top-width:2px;border-top-style:solid}
  .legend .dot{display:inline-block;width:8px;height:8px;border-radius:50%;border-width:1.4px;border-style:solid;background:#fff;margin-left:-14px}
  .figure{background:#fff;border:1px solid #d4dfed;border-radius:6px;padding:14px;overflow:auto;display:flex;justify-content:center}
  .figure svg{width:100%;height:auto;max-width:100%}
  #feature-list{max-height:150px;overflow:auto;display:grid;grid-template-columns:1fr 1fr;gap:2px 8px}
  .feat{display:flex;align-items:center;gap:5px;font-size:.72rem;color:#374e69}
  #stage{position:absolute;left:-99999px;top:0;visibility:hidden;pointer-events:none}
  .meta{margin-top:8px;color:#4a5f7d;font-size:.71rem}
  .note{margin-top:9px;padding:7px 9px;border-left:2px solid #245db4;background:#f1f6fd;
        color:#374e69;font-size:.71rem}
  @media (max-width:900px){.wrap{grid-template-columns:1fr;padding:12px}aside{order:2}}
</style>
</head>
<body>
<div class="page">
  <aside>
    <h2>Publication</h2>

    <details open>
      <summary>特征</summary>
      <div class="body" id="feature-list"></div>
    </details>

    <details open>
      <summary>画布与子图</summary>
      <div class="body">
        <div class="row"><span>子图宽 mm</span><input id="cellw" type="number" min="10" max="120" step="1" value="41"></div>
        <div class="row"><span>子图高 mm</span><input id="cellh" type="number" min="10" max="120" step="1" value="25"></div>
        <div class="row"><span>列间距 mm</span><input id="gapx" type="number" min="0" max="30" step="0.5" value="0"></div>
        <div class="row"><span>行间距 mm</span><input id="gapy" type="number" min="0" max="30" step="0.5" value="5"></div>
        <div class="row"><span>外边距 pt</span><input id="edgepad" type="number" min="0" max="12" step="0.5" value="1.5"></div>
        <div class="row"><span>自适应留白</span><input id="autocrop" type="checkbox" checked></div>
        <div class="row"><span>Rows × Cols</span>
          <span><input id="rows" type="number" min="1" max="10" value="3">
          <input id="cols" type="number" min="1" max="8" value="4"></span>
        </div>
      </div>
    </details>

    <details open>
      <summary>刻度与范围</summary>
      <div class="body">
        <div class="row"><span>X 刻度数</span><input id="xnt" type="number" min="2" max="8" value="3"></div>
        <div class="row"><span>Y 刻度数</span><input id="ynt" type="number" min="2" max="8" value="2"></div>
        <div class="row"><span>共享 Y 轴</span><input id="sharey" type="checkbox" checked></div>
        <div class="row"><span>X 左 / 右 边距 %</span>
          <span class="pair"><input id="mxl" type="number" min="0" max="80" step="1" value="20">
          <input id="mxr" type="number" min="0" max="80" step="1" value="20"></span>
        </div>
        <div class="row"><span>Y 下 / 上 边距 %</span>
          <span class="pair"><input id="myb" type="number" min="0" max="80" step="1" value="30">
          <input id="myt" type="number" min="0" max="80" step="1" value="10"></span>
        </div>
      </div>
    </details>

    <details>
      <summary>排版与轴标题</summary>
      <div class="body">
        <div class="row"><span>字体</span>
          <select id="font">
            <option value="Times New Roman, Times, serif" selected>Times</option>
            <option value="Helvetica, Arial, sans-serif">Helvetica / Arial</option>
            <option value="-apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif">系统 sans</option>
          </select>
        </div>
        <div class="row"><span>字号 pt</span><input id="fsize" type="number" min="4" max="14" step="0.5" value="11"></div>
        <div class="row"><span>面板字母</span><input id="letters" type="checkbox" checked></div>
        <div class="row"><span>网格线</span><input id="grid" type="checkbox"></div>
        <div class="row"><span>横轴标签</span><input id="xlabel" type="text" value="拟动态时间"></div>
        <div class="row"><span>纵轴标签</span><input id="ylabel" type="text" value="变换后的观测值"></div>
        <div class="row"><span>X 标题距子图 pt</span><input id="xoff" type="number" min="0" max="30" step="1" value="10"></div>
        <div class="row"><span>Y 标题距子图 pt</span><input id="yoff" type="number" min="0" max="40" step="1" value="0"></div>
      </div>
    </details>

    <details>
      <summary>图元</summary>
      <div class="body">
        <div class="row"><span>标记大小 pt</span><input id="msize" type="number" min="1" max="10" step="0.5" value="6.5"></div>
        <div class="row"><span>拟合线宽 pt</span><input id="lw" type="number" min="0.5" max="4" step="0.25" value="1.5"></div>
        <div class="row"><span>点色</span><input id="dcolor" type="color" value="#1f77b4"></div>
        <div class="row"><span>线色</span><input id="ccolor" type="color" value="#d62728"></div>
      </div>
    </details>

    <details>
      <summary>导出</summary>
      <div class="body">
        <div class="row"><span>PNG dpi</span>
          <select id="dpi"><option>300</option><option selected>600</option></select>
        </div>
        <button class="btn primary" id="svg">导出 SVG（物理尺寸）</button>
        <button class="btn" id="png">导出 PNG（按 dpi）</button>
        <div class="note">PDF/EPS 由服务端 matplotlib 渲染（fonttype 42 字体嵌入）；这些参数一一对应 figsize / gridspec / wspace / hspace / set_xlim / set_ylim。</div>
      </div>
    </details>
  </aside>

  <main>
    <div id="stage" aria-hidden="true"></div>
    <div class="legend" id="legend"></div>
    <div class="figure" id="figure"></div>
    <p class="meta" id="meta"></p>
  </main>
</div>
<script>
const DATA = __DATA__;
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

function settings() {
  return {
    cellWmm: Math.max(10, +$("cellw").value || 41),
    cellHmm: Math.max(10, +$("cellh").value || 25),
    gapXmm: Math.max(0, +$("gapx").value || 0),
    gapYmm: Math.max(0, +$("gapy").value || 0),
    edgePad: Math.max(0, Math.min(12, +$("edgepad").value || 0)),
    autocrop: $("autocrop").checked,
    rows: Math.max(1, Math.min(10, +$("rows").value || 3)),
    cols: Math.max(1, Math.min(8, +$("cols").value || 4)),
    xnt: Math.max(2, Math.min(8, +$("xnt").value || 3)),
    ynt: Math.max(2, Math.min(8, +$("ynt").value || 2)),
    sharey: $("sharey").checked,
    mxl: Math.max(0, +$("mxl").value || 0) / 100,
    mxr: Math.max(0, +$("mxr").value || 0) / 100,
    myb: Math.max(0, +$("myb").value || 0) / 100,
    myt: Math.max(0, +$("myt").value || 0) / 100,
    font: $("font").value,
    fsize: Math.max(4, Math.min(14, +$("fsize").value || 11)),
    letters: $("letters").checked,
    grid: $("grid").checked,
    xlabel: $("xlabel").value.trim(),
    ylabel: $("ylabel").value.trim(),
    xoff: Math.max(0, +$("xoff").value || 0),
    yoff: Math.max(0, +$("yoff").value || 0),
    msize: Math.max(1, Math.min(10, +$("msize").value || 6.5)),
    lw: Math.max(0.5, Math.min(4, +$("lw").value || 1.5)),
    dcolor: $("dcolor").value,
    ccolor: $("ccolor").value,
  };
}

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

  const cells = DATA.panels.filter(p => active.has(p.name)).slice(0, s.rows * s.cols);

  const seriesOf = p => p.series && p.series.length ? p.series : [{
    label: null, xs: p.xs, ys: p.ys, curveX: p.curveX, curveY: p.curveY,
    dcolor: s.dcolor, ccolor: s.ccolor,
  }];
  let xlo = Infinity, xhi = -Infinity, shared = [Infinity, -Infinity];
  const per = [];
  cells.forEach(p => {
    const list = seriesOf(p);
    let lo = Infinity, hi = -Infinity;
    list.forEach(ser => {
      const [a, b] = range(ser.curveX);
      xlo = Math.min(xlo, a); xhi = Math.max(xhi, b);
      const [c, d] = range(ser.ys.concat(ser.curveY));
      lo = Math.min(lo, c); hi = Math.max(hi, d);
    });
    per.push([lo, hi]);
    shared = [Math.min(shared[0], lo), Math.max(shared[1], hi)];
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
      if (row === s.rows - 1 && ti % skip === 0) {
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
    seriesOf(p).forEach(ser => {
      const line = ser.curveX.map((cx, k) =>
        `${k ? "L" : "M"}${x(cx).toFixed(2)},${y(ser.curveY[k]).toFixed(2)}`).join(" ");
      marks.appendChild(el("path", {d: line, fill: "none",
                                    stroke: ser.ccolor || s.ccolor,
                                    "stroke-width": s.lw, "stroke-linecap": "butt"}));
      ser.xs.forEach((cx, k) => marks.appendChild(el("circle", {
        cx: x(cx), cy: y(ser.ys[k]), r: s.msize / 2, fill: "none",
        stroke: ser.dcolor || s.dcolor, "stroke-width": s.lw * 0.75})));
    });
    svg.appendChild(marks);

    const cap = el("text", {x: ox + m.l + pw / 2, y: oy - 3, "text-anchor": "middle",
                            "font-size": s.fsize});
    cap.textContent = p.name;
    svg.appendChild(cap);
  });

  const gridLeft = pad.l + m.l;
  const gridRight = pad.l + (s.cols - 1) * (cellW + gapX) + m.l + (cellW - m.l - m.r);
  const gridTop = pad.t + titleRoom + m.t;
  const gridBottom = pad.t + titleRoom + (s.rows - 1) * (cellH + gapY) + m.t + (cellH - m.t - m.b);
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

["cellw", "cellh", "gapx", "gapy", "rows", "cols", "xnt", "ynt", "sharey",
 "mxl", "mxr", "myb", "myt", "font", "fsize", "letters", "grid", "xlabel", "ylabel",
 "xoff", "yoff", "msize", "lw", "dcolor", "ccolor", "edgepad", "autocrop"].forEach(id => {
  $(id).addEventListener("input", render);
  $(id).addEventListener("change", render);
});

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

const legend = document.getElementById("legend");
if (DATA.series && DATA.series.length) {
  DATA.series.forEach(entry => {
    const item = document.createElement("span");
    item.className = "item";
    const line = document.createElement("span");
    line.className = "swatch";
    line.style.borderTopColor = entry.ccolor;
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.borderColor = entry.dcolor;
    const label = document.createElement("span");
    label.textContent = entry.label;
    item.append(line, dot, label);
    legend.appendChild(item);
  });
} else {
  legend.remove();
}

const active = new Set(DATA.panels.map(p => p.name));
const featureList = document.getElementById("feature-list");
DATA.panels.forEach(p => {
  const row = document.createElement("label");
  row.className = "feat";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = true;
  box.addEventListener("change", () => {
    if (box.checked) active.add(p.name); else active.delete(p.name);
    render();
  });
  const span = document.createElement("span");
  span.textContent = p.name;
  row.append(box, span);
  featureList.appendChild(row);
});

render();
</script>
</body>
</html>
"""


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


COMPARE_SCATTER_COLORS = ("#F9B3AD", "#C9A1CA", "#76C2AF", "#E5C68F", "#C59FCE", "#A0D8E7")
COMPARE_CURVE_COLORS = ("#9E2223", "#A06EA5", "#52B793", "#DDB866", "#B488C2", "#79C6DF")


def build_compare_panels(scatter_frames, curve_frames, labels, features) -> list[dict[str, Any]]:
    """One panel per feature, one series per input file (markers + fitted curve)."""
    panels = []
    for name in features:
        series = []
        for i, (scatter, curve, label) in enumerate(zip(scatter_frames, curve_frames, labels)):
            series.append({
                "label": str(label),
                "xs": [round(float(v), 4) for v in scatter.index],
                "ys": [round(float(v), 4) for v in scatter[name]],
                "curveX": [round(float(v), 4) for v in curve.index],
                "curveY": [round(float(v), 4) for v in curve[name]],
                "dcolor": COMPARE_SCATTER_COLORS[i % len(COMPARE_SCATTER_COLORS)],
                "ccolor": COMPARE_CURVE_COLORS[i % len(COMPARE_CURVE_COLORS)],
            })
        panels.append({"name": str(name), "series": series})
    return panels


def render(panels: list[dict[str, Any]]) -> str:
    """Return a self-contained HTML document: control panel + figure + exports."""
    payload: dict[str, Any] = {"panels": panels}
    first = panels[0] if panels else {}
    if first.get("series"):
        payload["series"] = [
            {"label": s["label"], "dcolor": s["dcolor"], "ccolor": s["ccolor"]}
            for s in first["series"]
        ]
    return TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
