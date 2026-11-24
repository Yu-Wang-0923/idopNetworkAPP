"""Paper-style HTML/SVG renderer for FunClu cluster profiles.

Derived from the Curve Fitting publication renderer: points are authored in
points (physical mm sizing), the left panel holds collapsible publication
controls, and exports are vector SVG / DPI PNG. One panel per cluster; each
condition contributes member curves and a mean curve.
"""
from __future__ import annotations

import base64
import gzip
import json
import math
from typing import Any, Iterable

import numpy as np

CONDITION_COLORS = (
    ("#9E2223", "#F3C3C0"),
    ("#3060A8", "#BFD4EF"),
    ("#1E7F63", "#BFE6D8"),
    ("#8A5AA8", "#DFC9EC"),
    ("#A8721E", "#F0DDBB"),
    ("#2B7E9E", "#BEDFE9"),
)


def _round(values: Iterable[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def build_cluster_panels(
    curve_frames: dict[str, Any],
    labels: np.ndarray,
    common_cols: list[str],
    n_components: int,
    *,
    show_members: bool = True,
    show_ci: bool = False,
    max_members: int = 40,
) -> list[dict[str, Any]]:
    """Pack per-cluster series (mean, CI band, member curves) for the renderer."""
    labels = np.asarray(labels, dtype=int).ravel()
    if labels.size != len(common_cols):
        raise ValueError("labels do not match common columns")

    panels: list[dict[str, Any]] = []
    for k in range(n_components):
        member_idx = np.flatnonzero(labels == k)
        members = [common_cols[i] for i in member_idx]
        series = []
        for cond_order, (cond, frame) in enumerate(curve_frames.items()):
            cols = [c for c in members if c in frame.columns]
            if not cols:
                continue
            block = frame.loc[:, cols].astype(float)
            mean = block.mean(axis=1).to_numpy()
            if show_ci and len(cols) > 1:
                sem = block.std(axis=1, ddof=1).to_numpy() / math.sqrt(len(cols))
                lo = mean - 1.96 * sem
                hi = mean + 1.96 * sem
            else:
                lo = hi = None
            member_lines = []
            if show_members:
                step = max(1, len(cols) // max_members)
                for col in cols[::step][:max_members]:
                    member_lines.append(_round(frame.loc[:, col].astype(float).to_numpy()))
            color, band = CONDITION_COLORS[cond_order % len(CONDITION_COLORS)]
            series.append({
                "label": str(cond),
                "xs": [round(float(v), 4) for v in frame.index],
                "mean": _round(mean),
                "lo": _round(lo) if lo is not None else None,
                "hi": _round(hi) if hi is not None else None,
                "members": member_lines,
                "dcolor": color,
                "ccolor": color,
                "band": band,
            })
        panels.append({
            "name": f"M{k + 1}",
            "size": int(len(members)),
            "series": series,
        })
    return panels


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
  .mini{border:1px solid #bacde3;border-radius:4px;background:#fff;color:#1c304b;
        padding:0 6px;font-size:.8rem;line-height:1.5;cursor:pointer}
  .mini:disabled{opacity:.4;cursor:default}
  #clinfo{font-variant-numeric:tabular-nums;min-width:44px;text-align:center}
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
      <summary>布局</summary>
      <div class="body">
        <div class="row"><span>排列方式</span>
          <select id="layout">
            <option value="grid" selected>布局一：网格</option>
            <option value="col">布局二：每列 1 簇</option>
            <option value="row">布局三：每行 1 簇</option>
          </select>
        </div>
      </div>
    </details>

    <details open>
      <summary>簇</summary>
      <div class="body" id="cluster-list"></div>
      <div class="row" id="clpager"><span>绘图分页</span><span class="pair">
        <button type="button" id="clprev" class="mini">‹</button>
        <span id="clinfo">1 / 1</span>
        <button type="button" id="clnext" class="mini">›</button>
      </span></div>
    </details>

    <details open>
      <summary>画布与子图</summary>
      <div class="body">
        <div class="row"><span>子图宽 mm</span><input id="cellw" type="number" min="20" max="160" step="1" value="40"></div>
        <div class="row"><span>子图高 mm</span><input id="cellh" type="number" min="20" max="160" step="1" value="25"></div>
        <div class="row"><span>列间距 mm</span><input id="gapx" type="number" min="0" max="30" step="0.5" value="0"></div>
        <div class="row"><span>行间距 mm</span><input id="gapy" type="number" min="0" max="30" step="0.5" value="6"></div>
        <div class="row"><span>外边距 pt</span><input id="edgepad" type="number" min="0" max="12" step="0.5" value="0"></div>
        <div class="row"><span>自适应留白</span><input id="autocrop" type="checkbox" checked></div>
        <div class="row"><span>Rows × Cols</span>
          <span><input id="rows" type="number" min="1" max="10" value="2">
          <input id="cols" type="number" min="1" max="6" value="2"></span>
        </div>
      </div>
    </details>

    <details open>
      <summary>刻度与范围</summary>
      <div class="body">
        <div class="row"><span>X 刻度数</span><input id="xnt" type="number" min="2" max="8" value="3"></div>
        <div class="row"><span>Y 刻度数</span><input id="ynt" type="number" min="2" max="8" value="3"></div>
        <div class="row"><span>共享 Y 轴</span><input id="sharey" type="checkbox" checked></div>
        <div class="row"><span>X 左 / 右 边距 %</span>
          <span class="pair"><input id="mxl" type="number" min="0" max="40" step="1" value="10">
          <input id="mxr" type="number" min="0" max="40" step="1" value="10"></span>
        </div>
        <div class="row"><span>Y 下 / 上 边距 %</span>
          <span class="pair"><input id="myb" type="number" min="0" max="40" step="1" value="40">
          <input id="myt" type="number" min="0" max="40" step="1" value="30"></span>
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
        <div class="row"><span>横轴标签</span><input id="xlabel" type="text" value="Quasi-dynamic time"></div>
        <div class="row"><span>纵轴标签</span><input id="ylabel" type="text" value="Response"></div>
        <div class="row"><span>X 标题距子图 pt</span><input id="xoff" type="number" min="0" max="30" step="1" value="6"></div>
        <div class="row"><span>Y 标题距子图 pt</span><input id="yoff" type="number" min="0" max="40" step="1" value="4"></div>
      </div>
    </details>

    <details>
      <summary>图元</summary>
      <div class="body">
        <div class="row"><span>均值线宽 pt</span><input id="lw" type="number" min="0.5" max="4" step="0.25" value="1.5"></div>
        <div class="row"><span>成员线宽 pt</span><input id="memberlw" type="number" min="0.1" max="3" step="0.1" value="1"></div>
        <div class="row"><span>成员透明度 %</span><input id="memberalpha" type="number" min="5" max="80" step="5" value="20"></div>
        <div class="row"><span>成员曲线</span><input id="showmembers" type="checkbox" checked></div>
        <div class="row"><span>95% CI</span><input id="showci" type="checkbox"></div>
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
        <div class="note">PDF/EPS 由服务端 matplotlib 渲染；当前 SVG 可直接置入 AI/Inkscape。</div>
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
let DATA = null;
let active = null;
let clusterPage = 0;
async function loadData() {
  const packed = "__DATA_B64__";
  const raw = Uint8Array.from(atob(packed), c => c.charCodeAt(0));
  let text;
  if (typeof DecompressionStream === "function") {
    const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
    text = await new Response(stream).text();
  } else {
    document.body.innerHTML =
      '<p style="font:14px/1.5 sans-serif;color:#8a3b3b;padding:16px">' +
      'This browser cannot decompress the chart payload (need Chrome/Edge 80+ or Firefox 113+).</p>';
    return null;
  }
  return JSON.parse(text);
}
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
    layout: $("layout").value,
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
    msize: 6.5,
    memberLw: Math.max(0.1, Math.min(3, +$("memberlw").value || 0.6)),
    memberAlpha: Math.max(0.05, Math.min(0.8, (+$("memberalpha").value || 25) / 100)),
    members: $("showmembers").checked,
    ciOn: $("showci").checked,
    lw: Math.max(0.5, Math.min(4, +$("lw").value || 1.5)),
  };
}

const range = arr => arr.reduce((a, v) => [Math.min(a[0], v), Math.max(a[1], v)],
                                [Infinity, -Infinity]);

function buildSvg() {
  const s = settings();
  const visible = DATA.panels.filter(p => active.has(p.name));
  const conditions = (DATA.conditions && DATA.conditions.length) ? DATA.conditions : [];
  const nClusters = Math.max(1, visible.length);
  const nCond = Math.max(1, conditions.length);
  const effRows = s.layout === "row" ? nClusters : (s.layout === "col" ? nCond : s.rows);
  const effCols = s.layout === "col" ? nClusters : (s.layout === "row" ? nCond : s.cols);
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
  const W = pad.l + pad.r + effCols * cellW + (effCols - 1) * gapX;
  const H = pad.t + pad.b + effRows * cellH + (effRows - 1) * gapY + titleRoom;

  const pick = (p, cond) => (p.series || []).find(ser => ser.label === cond) || null;
  const perPage = Math.max(1, effRows * effCols);
const maxPage = Math.max(1, Math.ceil(visible.length / perPage));
if (clusterPage > maxPage - 1) clusterPage = maxPage - 1;
const pageCells = visible.slice(clusterPage * perPage, (clusterPage + 1) * perPage);
const cells = s.layout === "col"
    ? conditions.flatMap(cond => visible.map(p => ({p, cond, series: [pick(p, cond)].filter(Boolean)})))
    : s.layout === "row"
      ? visible.flatMap(p => conditions.map(cond => ({p, cond, series: [pick(p, cond)].filter(Boolean)})))
      : pageCells.map(p => ({p, cond: null, series: p.series || []}));
  const lastRow = Math.max(0, Math.ceil(cells.length / effCols) - 1);

  let xlo = Infinity, xhi = -Infinity, shared = [Infinity, -Infinity];
  cells.forEach(cell => {
    cell.series.forEach(ser => {
      const [a, b] = range(ser.xs);
      xlo = Math.min(xlo, a); xhi = Math.max(xhi, b);
      shared = [Math.min(shared[0], Math.min(...ser.mean)),
                Math.max(shared[1], Math.max(...ser.mean))];
    });
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

  cells.forEach((cell, i) => {
    const p = cell.p, cellSeries = cell.series;
    const row = Math.floor(i / effCols), col = i % effCols;
    const ox = pad.l + col * (cellW + gapX);
    const oy = pad.t + titleRoom + row * (cellH + gapY);
    const pw = cellW - m.l - m.r, ph = cellH - m.t - m.b;

    let ylo, yhi;
    if (s.sharey) [ylo, yhi] = sharedYs;
    else {
      const vals = cellSeries.flatMap(ser => ser.mean.concat(ser.lo || [], ser.hi || []));
      const [a, b] = range(vals), span = (b - a) || 1;
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
        const xDigits = (xMax - xMin) >= 10 ? 0 : (xMax - xMin) >= 1 ? 1 : 2;
        t.textContent = v.toFixed(xDigits);
        svg.appendChild(t);
      }
    });
    const showY = !s.sharey || col === 0;
    const ySpan = Math.abs(yhi - ylo);
    const yDigits = ySpan >= 10 ? 0 : ySpan >= 1 ? 1 : 2;
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
    cellSeries.forEach(ser => {
      const sx = ser.xs;
      if (s.ciOn && ser.lo && ser.hi) {
        const up = sx.map((cx, k) => `${x(cx).toFixed(2)},${y(ser.hi[k]).toFixed(2)}`);
        const dn = sx.map((cx, k) => `${x(cx).toFixed(2)},${y(ser.lo[k]).toFixed(2)}`).reverse();
        marks.appendChild(el("path", {
          d: `M${up.join(" L")} L${dn.join(" L")} Z`,
          fill: ser.band || "#dbe5f2", stroke: "none", opacity: 0.7}));
      }
      if (s.members) {
        ser.members.forEach(line => {
          const d = line.map((v, k) =>
            `${k ? "L" : "M"}${x(sx[k]).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
          marks.appendChild(el("path", {d, fill: "none", stroke: ser.dcolor || s.dcolor,
                                        "stroke-width": s.memberLw, opacity: s.memberAlpha}));
        });
      }
      const dm = ser.mean.map((v, k) =>
        `${k ? "L" : "M"}${x(sx[k]).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
      marks.appendChild(el("path", {d: dm, fill: "none", stroke: ser.dcolor || s.dcolor,
                                    "stroke-width": s.lw, "stroke-linecap": "round",
                                    "stroke-linejoin": "round"}));
    });
    svg.appendChild(marks);

    const cap = el("text", {x: ox + m.l + pw / 2, y: oy - 3, "text-anchor": "middle",
                            "font-size": s.fsize});
    cap.textContent = p.size ? `${p.name}  (n=${p.size})` : p.name;
    svg.appendChild(cap);
  });

  const gridLeft = pad.l + m.l;
  const gridRight = pad.l + (effCols - 1) * (cellW + gapX) + m.l + (cellW - m.l - m.r);
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
  const perNow = Math.max(1, settings().rows * settings().cols);
  const checkedNow = DATA.panels.filter(p => active.has(p.name)).length;
  const pagesNow = Math.max(1, Math.ceil(checkedNow / perNow));
  $("clinfo").textContent = `${Math.min(clusterPage + 1, pagesNow)} / ${pagesNow}`;
  $("clpager").style.display = settings().layout === "grid" ? "" : "none";
  $("clprev").disabled = clusterPage <= 0;
  $("clnext").disabled = clusterPage >= pagesNow - 1;
  $("meta").textContent =
    `画布 ${(W / PT_PER_MM).toFixed(0)} × ${(H / PT_PER_MM).toFixed(0)} mm · ` +
    `子图 ${(built.cellW / PT_PER_MM).toFixed(0)} × ${(built.cellH / PT_PER_MM).toFixed(0)} mm · ` +
    `${built.shown} 个面板 · ${DATA.panels.length} 簇 · ${DATA.conditions.length} 条件 · ` +
    (s.autocrop ? "自适应留白" : "固定留白");
}

["cellw", "cellh", "gapx", "gapy", "edgepad", "autocrop", "rows", "cols",
 "xnt", "ynt", "sharey", "mxl", "mxr", "myb", "myt", "font", "fsize",
 "letters", "grid", "xlabel", "ylabel", "xoff", "yoff", "lw",
 "memberlw", "memberalpha", "showmembers", "showci", "layout"].forEach(id => {
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

async function boot() {
  try {
    DATA = await loadData();
  } catch (err) {
    document.body.innerHTML =
      '<p style="font:14px/1.5 sans-serif;color:#8a3b3b;padding:16px">' +
      'Chart payload could not be decoded: ' + String(err) + '</p>';
    return;
  }
  if (!DATA) return;
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
active = new Set(DATA.panels.map(p => p.name));
const featureList = document.getElementById("cluster-list");
DATA.panels.forEach(p => {
  const row = document.createElement("label");
  row.className = "feat";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = true;
  box.addEventListener("change", () => {
    if (box.checked) active.add(p.name); else active.delete(p.name);
    clusterPage = 0;
    render();
  });
  const span = document.createElement("span");
  span.textContent = p.name;
  row.append(box, span);
  featureList.appendChild(row);
});

  $("clprev").onclick = () => { if (clusterPage > 0) { clusterPage -= 1; render(); } };
  $("clnext").onclick = () => { clusterPage += 1; render(); };
  render();
}
boot();
</script>
</body>
</html>
"""


def render(
    panels: list[dict[str, Any]],
    *,
    conditions: list[str],
    cols: int = 2,
    svg_label: str = "导出 SVG（物理尺寸）",
    png_label: str = "导出 PNG（按 dpi）",
) -> str:
    """Return a self-contained HTML document: publication panel + figure."""
    legend = []
    for i, cond in enumerate(conditions):
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)][0]
        legend.append({"label": str(cond), "dcolor": color, "ccolor": color})
    payload = {"panels": panels, "series": legend, "conditions": [str(c) for c in conditions]}
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packed = base64.b64encode(gzip.compress(raw, compresslevel=6)).decode("ascii")
    html = TEMPLATE.replace("__DATA_B64__", packed)
    cols = max(1, min(6, int(cols)))
    html = html.replace('id="cols" type="number" min="1" max="6" value="2"',
                        f'id="cols" type="number" min="1" max="6" value="{cols}"')
    return html
