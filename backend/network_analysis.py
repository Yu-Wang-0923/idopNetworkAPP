"""Network Analysis 后端：GLMY 同调（barcode）流程。

适配 ``pages/3_NetRecon.py`` Export 出的 ZIP（含 ``from_to.csv``，列为
``from, to, weight, type``），把权重输入 ``backend/GLMY.exe``，得到
``homology.json`` 后还原 +100 偏移并返回结构化结果。
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from backend.utils import BACKEND_DIR, GLMY_EXE_PATH


GLMY_WEIGHT_OFFSET: float = 100.0
# Linux 上首跑要预热 Wine prefix，给宽一些；Windows 仍用较短超时。
GLMY_TIMEOUT_SEC: int = 30 if sys.platform == "win32" else 120
WINE_PREFIX_DIR: Path = BACKEND_DIR.parent / ".wine_prefix"

NormalizeMode = Literal["auto", "raw"]


# ── ZIP 解析 ──────────────────────────────────────────────────────────────────

def _read_zip_member_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """从 ZIP 中读取 CSV 成员（兼容 utf-8-sig BOM）。"""
    with zf.open(name) as fh:
        return pd.read_csv(io.BytesIO(fh.read()))


def list_from_to_members(zip_bytes: bytes) -> list[str]:
    """枚举 ZIP 中所有 ``from_to.csv`` 路径。

    单层导出返回 ``["from_to.csv"]``；多层导出返回形如
    ``"inter_cluster/<cond>/from_to.csv"`` 与
    ``"intra_cluster/<cond>/<cluster>/from_to.csv"`` 的列表。
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.endswith("from_to.csv")]
    names.sort()
    return names


def member_display_label(member_path: str) -> str:
    """把 ZIP 内 from_to.csv 路径转成下拉框友好的标签。"""
    if member_path == "from_to.csv":
        return "single_layer"
    if member_path.endswith("/from_to.csv"):
        return member_path[: -len("/from_to.csv")]
    return member_path


def load_from_to_from_zip(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """从 ZIP 中读取指定 ``from_to.csv``，校验列名。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        df = _read_zip_member_csv(zf, member_path)
    required = {"from", "to", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{member_path} 缺少必需列: {sorted(missing)}；实际列={list(df.columns)}"
        )
    df = df.copy()
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"]).reset_index(drop=True)
    return df


# ── GLMY 输入构造 ────────────────────────────────────────────────────────────

def build_glmy_input(
    from_to_df: pd.DataFrame,
    *,
    normalize: NormalizeMode = "auto",
    weight_offset: float = GLMY_WEIGHT_OFFSET,
) -> tuple[str, dict[str, int], int, float]:
    """把 from_to 表构造为 GLMY.exe 的 stdin 字符串。

    GLMY.exe 要求**权重为正**，且对量级敏感（原 ``GLMY1.py`` 的 +100 偏移
    是按 [-1, 1] 量级的相关系数设计的）。当 ``weight`` 量级远超 1（如几百到
    几千）时，``weight + 100`` 仍可能为负或者 offset 完全失效，导致 GLMY 输出
    空 homology。本函数支持先把权重缩放到 [-1, 1] 再加 offset。

    Parameters
    ----------
    from_to_df: 含 ``from``/``to``/``weight`` 三列的 DataFrame。
    normalize:
        - ``"auto"``（默认）: ``scale = max(|weight|, eps)``；GLMY 输入用
          ``weight / scale + offset``。
        - ``"raw"``: 不缩放，直接 ``weight + offset``（适合权重已在 [-1, 1] 量级时）。
    weight_offset: 加在 (缩放后) weight 上的偏移，确保 GLMY 收到的全是正数。

    Returns
    -------
    input_str, vertex_id_map, n_vertices_plus_one, scale_factor
        ``scale_factor`` 表示「缩放因子」：``auto`` 模式下等于 ``max(|weight|)``，
        ``raw`` 模式下恒为 1.0。后续 ``_strip_offset_from_homology`` 会乘回这个
        因子，让 barcode 的 birth/death 仍处在原始 weight 尺度。

        ``input_str`` 形如::

            v1,v2,...,vN
            (u,v,w)
            ...
            #
            4
            y
    """
    if from_to_df.empty:
        raise ValueError("from_to.csv 为空，无法运行 GLMY。")

    vertices = sorted(set(from_to_df["from"]).union(set(from_to_df["to"])))
    vertex_id_map = {name: idx + 1 for idx, name in enumerate(vertices)}
    input_v = ",".join(str(vertex_id_map[name]) for name in vertices)

    if normalize == "auto":
        abs_max = float(from_to_df["weight"].abs().max())
        # eps 防 0；权重全为 0 时退化为 raw 模式（scale=1）
        scale_factor = abs_max if abs_max > 1e-12 else 1.0
    elif normalize == "raw":
        scale_factor = 1.0
    else:  # pragma: no cover - 类型校验
        raise ValueError(f"未知 normalize 模式: {normalize!r}")

    edges_lines: list[str] = []
    for _, row in from_to_df.iterrows():
        u = vertex_id_map[str(row["from"])]
        v = vertex_id_map[str(row["to"])]
        w = float(row["weight"]) / scale_factor + weight_offset
        edges_lines.append(f"({u},{v},{w})")

    input_str = input_v + "\n" + "\n".join(edges_lines) + "\n#\n4\ny"
    return input_str, vertex_id_map, len(vertices) + 1, scale_factor


# ── GLMY.exe 调用 ────────────────────────────────────────────────────────────

def _strip_offset_from_homology(
    homology: dict[str, list[list[float]]],
    *,
    weight_offset: float,
    scale_factor: float = 1.0,
) -> dict[str, list[list[float]]]:
    """把 homology.json 中 birth/death 还原到原始 weight 尺度。

    流程：``(value - weight_offset) * scale_factor``。``-1`` 代表无穷区间，
    保持原值。
    """
    processed: dict[str, list[list[float]]] = {}
    for key, value_list in homology.items():
        new_list: list[list[float]] = []
        for sublist in value_list:
            new_sub: list[float] = []
            for item in sublist:
                if item != -1 and isinstance(item, (int, float)):
                    restored = (float(item) - weight_offset) * scale_factor
                    new_sub.append(round(restored, 6))
                else:
                    new_sub.append(item)
            new_list.append(new_sub)
        processed[key] = new_list
    return processed


def _resolve_wine_prefix() -> str:
    """选一个可写目录作 ``WINEPREFIX``：仓库内优先，失败则退回 ``$HOME/.wine_prefix``。"""
    candidates: list[Path] = [WINE_PREFIX_DIR]
    home = os.environ.get("HOME")
    if home:
        candidates.append(Path(home) / ".wine_prefix")
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return str(path)
        except OSError:
            continue
    return tempfile.mkdtemp(prefix="wineprefix_")


def _build_glmy_command(exe: Path) -> tuple[list[str], dict[str, str]]:
    """根据当前平台决定如何执行 GLMY.exe，并准备子进程环境变量。

    - Windows: 直接 ``[exe]``。
    - 其它平台: 用 ``xvfb-run -a wine exe`` 包一层；若 ``xvfb-run`` 缺失则回退
      ``wine exe``。同时注入 ``WINEPREFIX`` / ``WINEDEBUG`` 等环境变量。
    """
    if sys.platform == "win32":
        return [str(exe)], dict(os.environ)

    if shutil.which("wine") is None:
        raise FileNotFoundError(
            "Wine 未安装。修复步骤（按顺序操作）：\n"
            "  1. 确认仓库根有 `packages.txt`，内容包含 `wine` 与 `xvfb`；\n"
            "  2. `git push` 把 `packages.txt` 推到远端（**仅修改不推送 Cloud 拿不到**）；\n"
            "  3. 在 Streamlit Community Cloud → 你的 App → Manage app → "
            "**Reboot app**（关键：rerun 不会触发 apt，只有 Reboot 会重装系统包）；\n"
            "  4. 等 Build 日志出现 `apt-get install ... wine ...`，再回页面 Run GLMY；\n"
            "  5. 若 Reboot 后日志报 `Unable to locate package wine`，说明镜像默认 apt "
            "源里没有 wine —— 把这条日志贴出来，我们换 `wine64` / `wine-stable` 之类的包名。"
        )

    cmd: list[str] = []
    if shutil.which("xvfb-run") is not None:
        cmd.extend(["xvfb-run", "-a"])
    cmd.extend(["wine", str(exe)])

    env = dict(os.environ)
    env["WINEPREFIX"] = _resolve_wine_prefix()
    env.setdefault("WINEDEBUG", "-all")
    env.setdefault("WINEDLLOVERRIDES", "mscoree=d;mshtml=d")
    env.setdefault("DISPLAY", ":0")
    return cmd, env


def run_glmy(
    from_to_df: pd.DataFrame,
    *,
    exe_path: Path | str | None = None,
    normalize: NormalizeMode = "auto",
    weight_offset: float = GLMY_WEIGHT_OFFSET,
    timeout: int = GLMY_TIMEOUT_SEC,
) -> dict[str, Any]:
    """调用 GLMY.exe 并解析 ``homology.json``。

    - **Windows**：直接 ``subprocess.run([exe])``；
    - **Linux / macOS（如 Streamlit Cloud 容器）**：通过 ``xvfb-run -a wine`` 调用，
      ``WINEPREFIX`` 优先放仓库内 ``.wine_prefix``，`packages.txt` 中需声明
      ``wine`` 与 ``xvfb``。

    每次调用在独立的临时目录里运行（GLMY.exe 在 cwd 写出 ``homology.json``），
    避免污染服务进程工作目录。

    Returns
    -------
    dict 包含::

        {
            "homology_raw": {...},          # 原始 birth/death（含 +offset）
            "homology": {...},              # 已减去 offset 的 birth/death
            "stdout": str,
            "stderr": str,
            "returncode": int,
            "vertex_id_map": {name: id},
            "input_str": str,
            "command": [...],               # 实际执行命令，用于排错
        }
    """
    exe = Path(exe_path) if exe_path is not None else GLMY_EXE_PATH
    if not exe.exists():
        raise FileNotFoundError(f"未找到 GLMY 可执行文件: {exe}")

    if sys.platform != "win32":
        try:
            os.chmod(exe, 0o755)
        except OSError:
            pass

    input_str, vertex_id_map, _, scale_factor = build_glmy_input(
        from_to_df, normalize=normalize, weight_offset=weight_offset
    )

    command, env = _build_glmy_command(exe)

    with tempfile.TemporaryDirectory(prefix="glmy_") as tmpdir:
        tmp_path = Path(tmpdir)
        try:
            proc = subprocess.run(
                command,
                input=input_str,
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=env,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"无法启动 GLMY 调用链 {command!r}：{e}。\n"
                "如部署在 Streamlit Community Cloud，请确认 `packages.txt` 含 "
                "`wine` 与 `xvfb`，并 Reboot app 让 apt 安装系统依赖。"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"GLMY 执行超过 {timeout}s 未返回（命令: {command!r}）。"
            ) from e

        homology_path = tmp_path / "homology.json"
        if not homology_path.exists():
            raise RuntimeError(
                "GLMY 未生成 homology.json。\n"
                f"command={command!r}\n"
                f"returncode={proc.returncode}\n"
                f"stdout=\n{proc.stdout}\n"
                f"stderr=\n{proc.stderr}"
            )

        with homology_path.open("r", encoding="utf-8") as f:
            homology_raw = json.load(f)

    homology = _strip_offset_from_homology(
        homology_raw,
        weight_offset=weight_offset,
        scale_factor=scale_factor,
    )

    return {
        "homology_raw": homology_raw,
        "homology": homology,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "vertex_id_map": vertex_id_map,
        "input_str": input_str,
        "command": command,
        "normalize": normalize,
        "scale_factor": scale_factor,
    }


# ── 工具：自适应 max_x ────────────────────────────────────────────────────────

def suggest_max_x(
    from_to_df: pd.DataFrame,
    *,
    buffer_ratio: float = 0.1,
    floor: float = 1.0,
) -> float:
    """根据 ``weight`` 绝对值最大值估计 barcode 横轴范围。"""
    if from_to_df.empty:
        return floor
    abs_max = float(from_to_df["weight"].abs().max())
    suggested = abs_max * (1.0 + max(0.0, buffer_ratio))
    return max(floor, suggested)


# ── 工具：合法文件名片段 ─────────────────────────────────────────────────────

def sanitize_name(name: str) -> str:
    """把任意字符串规范成可安全用于文件名的片段。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "glmy"
