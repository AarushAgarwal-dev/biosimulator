"""Generate BioSimulateAI researcher explainer videos from real model outputs.

Usage:
    python tools/make_videos.py [--only pipeline|bmp|graph]

Frames are rendered by matplotlib's Agg backend and streamed as raw RGB to
ffmpeg.  No intermediate frame directory is created.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Arc, Ellipse, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bmp_embryo
import nl_compiler
from simulation_engine import ODEModel, derive_edges_from_odes

W, H, DPI, FPS = 1280, 720, 100, 30
OUT = ROOT / "static" / "videos"
BG = "#0b0f1a"
PANEL = "#111827"
PANEL_2 = "#172033"
TEAL = "#00f2fe"
VIOLET = "#8a2be2"
GREEN = "#00ff87"
PINK = "#ff007f"
TEXT = "#f3f4f6"
MUTED = "#9ca3af"
AMBER = "#fbbf24"
GRID = "#263247"

PIPELINE_TEXT = (
    "p53 activates Mdm2 production. Mdm2 inhibits p53 production. "
    "p53 starts at 1."
)

VIDEO_META = {
    "pipeline": {
        "id": "pipeline",
        "title": "From plain language to a runnable model",
        "description": "A deterministic description-to-equations workflow using the app's compiler, equation-derived graph, solver, and semantic verification.",
        "file": "pipeline.mp4",
        "poster": "pipeline-poster.png",
        "captions": "pipeline.vtt",
        "duration_s": 42.0,
        "source_note": "Generated from nl_compiler.compile_text on the displayed p53/Mdm2 text, bp['_process_table'], bp['odes']/bp['fluxes'], simulation_engine.derive_edges_from_odes, ODEModel.simulate, and bp['_verification'].",
    },
    "bmp": {
        "id": "bmp_embryo",
        "title": "BMP shuttling patterns the Drosophila embryo",
        "description": "The Umulis et al. 2010 SBP mechanism: the solver is verified against the fully specified 2006 predecessor, then the 2010 model forms the contracting dorsal BMP stripe; every check is shown with its measured numbers.",
        "file": "bmp_embryo.mp4",
        "poster": "bmp_embryo-poster.png",
        "captions": "bmp_embryo.vtt",
        "duration_s": 48.0,
        "source_note": "Generated from bmp_embryo.simulate_surface(wt, SBP, 40x24, 0-60 min), simulate_cross_section(wt at 20/30/40/60 min; sog+/-, sog-/- and tld-/- at 60 min) with the default umulis2010 parameter set, and static/data/bmp/validation.json.",
    },
    "graph": {
        "id": "graph_guide",
        "title": "Reading the interaction graph",
        "description": "A precise guide to equation-derived edge signs, self-regulation, turnover omission, and the 5% readability threshold.",
        "file": "graph_guide.mp4",
        "poster": "graph_guide-poster.png",
        "captions": "graph_guide.vtt",
        "duration_s": 40.0,
        "source_note": "Generated from a four-species ODE blueprint evaluated by simulation_engine.ODEModel and derive_edges_from_odes; edge signs and strengths are Jacobian partial derivatives at the displayed initial state.",
    },
}

CAPTIONS = {
    "pipeline": [
        (0, 5, "From plain language to a runnable model."),
        (5, 11, "The deterministic compiler identifies p53 and Mdm2 as species."),
        (11, 18, "Each supported statement becomes a process card with its evidence quote."),
        (18, 26, "The process fluxes assemble into explicit ordinary differential equations."),
        (26, 32, "The interaction graph is derived from those equations, not drawn independently."),
        (32, 38, "The app integrates the equations to obtain p53 and Mdm2 trajectories."),
        (38, 42, "Coverage, Jacobian signs, and numerical simulation all pass verification."),
    ],
    "bmp": [
        (0, 5, "BMP shuttling patterns the Drosophila embryo, following Umulis and colleagues, 2010."),
        (5, 13, "Dorsal Dpp and Tsg oppose lateral Sog. Sog-Tsg carries BMP toward the dorsal midline, where Tld cleavage releases it."),
        (13, 25, "The surface solution: receptor-bound BMP rises from zero and concentrates on the dorsal side over sixty minutes."),
        (25, 34, "The cross-section contracts: the midline rises while the flanks lose signal. Dots mark the published Figure 4F midline values."),
        (34, 42, "At sixty minutes: sog heterozygotes are wider, sog nulls broad and weak, tld nulls have essentially no signal."),
        (42, 48, "The solver reproduces the fully specified 2006 model; the checks that pass and those that do not are both reported."),
    ],
    "graph": [
        (0, 5, "Reading the interaction graph."),
        (5, 12, "An arrow means the source species appears in the target species rate of change."),
        (12, 18, "A positive partial derivative is green activation; a negative partial derivative is pink inhibition."),
        (18, 25, "A self-arrow means autoregulation. Pure first-order turnover, minus k times X, is not drawn as a self-edge."),
        (25, 33, "Influences below five percent of the strongest influence on a target are folded from the readable view."),
        (33, 37, "Show all connections reveals folded influences as thin dashed edges."),
        (37, 40, "The graph is a readable view. The Equations tab is always complete."),
    ],
}


def ease(x: float) -> float:
    """Cubic ease-in-out on [0, 1]."""
    x = float(np.clip(x, 0.0, 1.0))
    return 3.0 * x * x - 2.0 * x * x * x


def ramp(t: float, start: float, end: float) -> float:
    if end <= start:
        return float(t >= end)
    return ease((t - start) / (end - start))


def scene_alpha(t: float, start: float, end: float, fade: float = 0.55) -> float:
    return ramp(t, start, start + fade) * (1.0 - ramp(t, end - fade, end))


def mix(c1: str, c2: str, a: float) -> tuple[float, float, float]:
    p, q = np.asarray(mcolors.to_rgb(c1)), np.asarray(mcolors.to_rgb(c2))
    return tuple((1 - a) * p + a * q)


def base_axes(fig: plt.Figure):
    fig.clear()
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)
    return ax


def label(ax, x, y, s, size=22, color=TEXT, weight="normal", ha="left", va="center", alpha=1.0, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontfamily="DejaVu Sans",
                   fontweight=weight, ha=ha, va=va, alpha=alpha, clip_on=True, **kw)


def wrapped(ax, x, y, s, width=55, size=20, color=TEXT, weight="normal", ha="left", va="top", alpha=1.0, linespacing=1.25, **kw):
    return label(ax, x, y, "\n".join(textwrap.wrap(str(s), width=width)), size=size,
                 color=color, weight=weight, ha=ha, va=va, alpha=alpha,
                 linespacing=linespacing, **kw)


def rounded(ax, x, y, w, h, face=PANEL, edge=GRID, radius=0.018, alpha=1.0, lw=1.5):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                           facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha)
    ax.add_patch(patch)
    return patch


def header(ax, kicker: str, title: str, subtitle: str | None = None, alpha=1.0):
    label(ax, 0.055, 0.935, kicker.upper(), 14, TEAL, "bold", alpha=alpha)
    label(ax, 0.055, 0.885, title, 31, TEXT, "bold", alpha=alpha)
    if subtitle:
        label(ax, 0.055, 0.838, subtitle, 16, MUTED, alpha=alpha)
    ax.plot([0.055, 0.945], [0.81, 0.81], color=GRID, lw=1.2, alpha=alpha)


def chip(ax, x, y, text, color=TEAL, alpha=1.0, width=None):
    w = width or max(0.085, 0.018 + len(text) * 0.011)
    rounded(ax, x, y - 0.026, w, 0.052, face=mix(BG, color, 0.12), edge=color, radius=0.025, alpha=alpha)
    label(ax, x + w / 2, y, text, 16, color, "bold", ha="center", alpha=alpha)


def draw_arrow(ax, start, end, color=GREEN, progress=1.0, inhibition=False,
               alpha=1.0, lw=3.0, dashed=False, curve=0.0):
    progress = ease(progress)
    sx, sy = start
    ex, ey = end
    px, py = sx + (ex - sx) * progress, sy + (ey - sy) * progress
    ls = "--" if dashed else "-"
    if inhibition:
        arr = FancyArrowPatch((sx, sy), (px, py), arrowstyle="-", connectionstyle=f"arc3,rad={curve}",
                              color=color, lw=lw, linestyle=ls, alpha=alpha)
        ax.add_patch(arr)
        if progress > 0.95:
            angle = math.atan2(ey - sy, ex - sx) + math.pi / 2
            dx, dy = 0.014 * math.cos(angle), 0.014 * math.sin(angle)
            ax.plot([ex - dx, ex + dx], [ey - dy, ey + dy], color=color, lw=lw, alpha=alpha)
    else:
        arr = FancyArrowPatch((sx, sy), (px, py), arrowstyle="-|>", mutation_scale=17,
                              connectionstyle=f"arc3,rad={curve}", color=color, lw=lw,
                              linestyle=ls, alpha=alpha, shrinkA=0, shrinkB=0)
        ax.add_patch(arr)


def node(ax, xy, text, color=VIOLET, alpha=1.0, radius=0.052):
    ax.add_patch(Circle(xy, radius, facecolor=mix(BG, color, 0.24), edgecolor=color,
                        linewidth=2.2, alpha=alpha))
    label(ax, xy[0], xy[1], text, 16, TEXT, "bold", ha="center", alpha=alpha)


def timecode(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_vtt(key: str):
    lines = ["WEBVTT", ""]
    for i, (start, end, text) in enumerate(CAPTIONS[key], 1):
        lines.extend([str(i), f"{timecode(start)} --> {timecode(end)}", text, ""])
    (OUT / VIDEO_META[key]["captions"]).write_text("\n".join(lines), encoding="utf-8")


def ffmpeg_command(path: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "26",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]


def canvas_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(rgba[:, :, :3])


def encode_video(key: str, renderer: Callable[[plt.Figure, float], None]):
    meta = VIDEO_META[key]
    duration = float(meta["duration_s"])
    frames = int(round(duration * FPS))
    path = OUT / meta["file"]
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    proc = subprocess.Popen(ffmpeg_command(path), stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for i in range(frames):
            renderer(fig, i / FPS)
            proc.stdin.write(canvas_rgb(fig).tobytes())
            if (i + 1) % (FPS * 10) == 0:
                print(f"  {key}: {(i + 1) / FPS:.0f}/{duration:.0f} s", flush=True)
        proc.stdin.close()
        code = proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed for {path.name} with exit code {code}")
    except Exception:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.kill()
        proc.wait()
        raise
    finally:
        plt.close(fig)

    poster_fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    renderer(poster_fig, {"pipeline": 28.0, "bmp": 19.0, "graph": 14.0}[key])
    poster_fig.savefig(OUT / meta["poster"], dpi=DPI, facecolor=BG)
    plt.close(poster_fig)
    write_vtt(key)


def load_pipeline_data() -> dict[str, Any]:
    bp = nl_compiler.compile_text(PIPELINE_TEXT, None)
    if bp.get("validation_errors"):
        raise RuntimeError("Pipeline example failed deterministic compilation: " + "; ".join(bp["validation_errors"]))
    model = ODEModel(bp)
    sim = model.simulate(float(bp["simulation_config"]["t_max"]), 241)
    edges = derive_edges_from_odes(bp)
    if not edges:
        raise RuntimeError("Pipeline equations produced no derived graph edges")
    return {"bp": bp, "model": model, "sim": sim, "edges": edges,
            "latex": model.get_equations_latex(verbose=False)}


def pipeline_renderer(data: dict[str, Any]) -> Callable[[plt.Figure, float], None]:
    bp, sim, edges, latex = data["bp"], data["sim"], data["edges"], data["latex"]
    process_rows = bp["_process_table"]

    def render(fig: plt.Figure, t: float):
        ax = base_axes(fig)
        # Scene 1: type the real input.
        if t < 6.0:
            a = scene_alpha(t, 0, 6)
            header(ax, "BioSimulateAI workflow", "From plain language to a runnable model", alpha=a)
            rounded(ax, 0.075, 0.28, 0.85, 0.40, face=PANEL, edge=VIOLET, alpha=a)
            label(ax, 0.105, 0.625, "MODEL DESCRIPTION", 13, MUTED, "bold", alpha=a)
            visible = int(len(PIPELINE_TEXT) * ramp(t, 0.8, 4.8))
            wrapped(ax, 0.105, 0.555, PIPELINE_TEXT[:visible], width=52, size=24, alpha=a)
            cursor = 0.25 + 0.5 * (0.5 + 0.5 * math.sin(t * 8))
            label(ax, 0.105, 0.345, "Deterministic rules path • no LLM", 15, TEAL, "bold", alpha=a * cursor)
        # Scene 2: species chips.
        if 5.4 <= t < 12.0:
            a = scene_alpha(t, 5.4, 12)
            header(ax, "1 · Parse", "Identify represented species", "Exact names become simulation state variables", alpha=a)
            rounded(ax, 0.07, 0.51, 0.86, 0.19, face=PANEL, edge=GRID, alpha=a)
            wrapped(ax, 0.10, 0.65, PIPELINE_TEXT, width=78, size=21, alpha=a)
            p = ramp(t, 6.0, 7.6)
            chip(ax, 0.25 - 0.06 * (1-p), 0.39, "p53", TEAL, a * p)
            chip(ax, 0.55 + 0.06 * (1-p), 0.39, "Mdm2", VIOLET, a * p)
            label(ax, 0.50, 0.245, "2 species • initial p53 = 1.0 • initial Mdm2 = 0.1", 18, MUTED, ha="center", alpha=a * ramp(t, 7.2, 8.4))
        # Scene 3: real process table cards.
        if 11.4 <= t < 19.0:
            a = scene_alpha(t, 11.4, 19)
            header(ax, "2 · Compile", "Statements become traceable processes", "Every card retains its evidence quote", alpha=a)
            positions = [(0.07, 0.54), (0.52, 0.54), (0.07, 0.27), (0.52, 0.27)]
            for i, (row, (x, y)) in enumerate(zip(process_rows, positions)):
                pa = a * ramp(t, 12.0 + i * 0.55, 12.7 + i * 0.55)
                edge = GREEN if "activated" in row else PINK if "repressed" in row else VIOLET
                rounded(ax, x, y, 0.40, 0.18, face=PANEL, edge=edge, alpha=pa)
                title, quote = row.split(" — ", 1)
                wrapped(ax, x + 0.025, y + 0.145, title, width=39, size=15, weight="bold", alpha=pa)
                wrapped(ax, x + 0.025, y + 0.055, quote, width=45, size=12, color=MUTED, alpha=pa)
        # Scene 4: equations assembled from the actual blueprint/model.
        if 18.4 <= t < 27.0:
            a = scene_alpha(t, 18.4, 27)
            header(ax, "3 · Assemble", "Processes become explicit rate laws", "Named fluxes are substituted into the ODEs", alpha=a)
            for i, sid in enumerate(["p53", "Mdm2"]):
                y = 0.59 - i * 0.24
                pa = a * ramp(t, 19.2 + i * 0.8, 20.3 + i * 0.8)
                rounded(ax, 0.09, y - 0.08, 0.82, 0.17, face=PANEL, edge=TEAL if i == 0 else VIOLET, alpha=pa)
                try:
                    label(ax, 0.50, y + 0.01, f"${latex[sid]}$", 19, TEXT, ha="center", alpha=pa)
                except Exception:
                    label(ax, 0.12, y + 0.01, f"d[{sid}]/dt = {bp['odes'][sid]}", 17, TEXT, alpha=pa)
            label(ax, 0.50, 0.185, f"{len(bp['fluxes'])} named fluxes • {len(bp['odes'])} coupled equations", 17, MUTED, ha="center", alpha=a)
        # Scene 5: equation-derived graph.
        if 26.4 <= t < 33.0:
            a = scene_alpha(t, 26.4, 33)
            header(ax, "4 · Derive", "The graph comes from the equations", "Sign = sign of the Jacobian partial derivative", alpha=a)
            pos = {"p53": (0.31, 0.46), "Mdm2": (0.69, 0.46)}
            node(ax, pos["p53"], "p53", TEAL, a)
            node(ax, pos["Mdm2"], "Mdm2", VIOLET, a)
            for i, e in enumerate(edges):
                progress = ramp(t, 27.3 + i * 0.7, 28.5 + i * 0.7)
                if e["source"] == "p53":
                    draw_arrow(ax, (0.37, 0.49), (0.63, 0.49), GREEN, progress, False, a, curve=0.15)
                    label(ax, 0.50, 0.61, "∂(dMdm2/dt)/∂p53 > 0", 17, GREEN, "bold", ha="center", alpha=a * progress)
                else:
                    draw_arrow(ax, (0.63, 0.43), (0.37, 0.43), PINK, progress, True, a, curve=0.15)
                    label(ax, 0.50, 0.32, "∂(dp53/dt)/∂Mdm2 < 0", 17, PINK, "bold", ha="center", alpha=a * progress)
            label(ax, 0.50, 0.19, "Green arrow: activation     Pink tee: inhibition", 16, MUTED, ha="center", alpha=a)
        # Scene 6: real trajectory draw.
        if 32.4 <= t < 38.8:
            a = scene_alpha(t, 32.4, 38.8)
            header(ax, "5 · Simulate", "Integrate the runnable model", "SciPy LSODA • rtol 1e-8 • atol 1e-10", alpha=a)
            plot = fig.add_axes([0.11, 0.18, 0.78, 0.55], facecolor=PANEL)
            for spine in plot.spines.values(): spine.set_color(GRID)
            plot.tick_params(colors=MUTED, labelsize=11)
            plot.grid(color=GRID, lw=0.7, alpha=0.65)
            tt = np.asarray(sim["t"])
            n = max(2, int(len(tt) * ramp(t, 33.0, 37.0)))
            plot.plot(tt[:n], np.asarray(sim["species"]["p53"])[:n], color=TEAL, lw=3, label="p53")
            plot.plot(tt[:n], np.asarray(sim["species"]["Mdm2"])[:n], color=VIOLET, lw=3, label="Mdm2")
            plot.set_xlim(0, tt[-1]); plot.set_ylim(0, 5.5)
            plot.set_xlabel("time", color=MUTED, fontsize=12); plot.set_ylabel("concentration", color=MUTED, fontsize=12)
            leg = plot.legend(loc="upper left", frameon=False, fontsize=12)
            for tx in leg.get_texts(): tx.set_color(TEXT)
            plot.patch.set_alpha(a)
            for artist in plot.get_children():
                if hasattr(artist, "set_alpha") and artist not in plot.spines.values():
                    try: artist.set_alpha(a)
                    except Exception: pass
        # Scene 7: verification closing card.
        if t >= 38.2:
            a = scene_alpha(t, 38.2, 42.01, fade=0.45)
            header(ax, "6 · Verify", "The blueprint passes semantic checks", "Real report from bp['_verification']", alpha=a)
            ver = bp["_verification"]
            checks = [
                ("Entity coverage", len(ver["coverage"]["missing"]) == 0, "0 named entities missing"),
                ("Jacobian sign checks", all(x["ok"] for x in ver["sign_checks"]), f"{len(ver['sign_checks'])}/{len(ver['sign_checks'])} passed"),
                ("Numerical simulation", ver["simulation"]["success"], f"finite, bounded to t = {ver['simulation']['end_time']:.0f}"),
            ]
            for i, (name, ok, detail) in enumerate(checks):
                y = 0.64 - i * 0.16
                rounded(ax, 0.14, y - 0.055, 0.72, 0.11, face=PANEL, edge=GREEN if ok else PINK, alpha=a)
                label(ax, 0.18, y, "✓" if ok else "×", 24, GREEN if ok else PINK, "bold", alpha=a)
                label(ax, 0.23, y + 0.018, name, 17, TEXT, "bold", alpha=a)
                label(ax, 0.23, y - 0.023, detail, 13, MUTED, alpha=a)
            label(ax, 0.50, 0.115, "Description → IR → equations → graph → trajectories → verification", 16, TEAL, "bold", ha="center", alpha=a)
    return render


def load_bmp_data() -> dict[str, Any]:
    # Real solves with the default (documented) umulis2010 parameter set.
    surface = bmp_embryo.simulate_surface(
        mechanism="sbp", perturbation="wt", nu=40, nv=24,
        save_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], time_budget_s=120.0,
    )
    cross = bmp_embryo.simulate_cross_section(
        mechanism="sbp", perturbation="wt", n_nodes=57,
        save_times=[15.0, 20.0, 30.0, 40.0, 45.0, 60.0],
    )
    sog = bmp_embryo.simulate_cross_section(
        mechanism="sbp", perturbation="sog_het", n_nodes=57, save_times=[60.0],
    )
    sog_null = bmp_embryo.simulate_cross_section(
        mechanism="sbp", perturbation="sog_null", n_nodes=57, save_times=[60.0],
    )
    tld = bmp_embryo.simulate_cross_section(
        mechanism="sbp", perturbation="tld_null", n_nodes=57, save_times=[60.0],
    )
    validation_path = ROOT / "static" / "data" / "bmp" / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else bmp_embryo.validate(include_surface=True)
    return {"surface": surface, "cross": cross, "sog": sog, "sog_null": sog_null, "tld": tld, "validation": validation}


def bmp_renderer(data: dict[str, Any]) -> Callable[[plt.Figure, float], None]:
    surface, cross, sog, sog_null, tld, validation = (data[k] for k in ("surface", "cross", "sog", "sog_null", "tld", "validation"))
    st = np.asarray(surface["times"], float)
    brs = np.asarray(surface["fields"]["BR"], float)
    x = np.asarray(surface["grid"]["x_um"], float)
    y = np.asarray(surface["grid"]["y_um"], float)
    z = np.asarray(surface["grid"]["z_um"], float)
    vmax = max(float(np.max(brs)), 1e-12)
    grid_label = f"{surface['grid']['nu']}×{surface['grid']['nv']} surface grid"
    gu = np.asarray(surface["grid"]["u"], float)
    gv = np.asarray(surface["grid"]["v"], float)
    A_um, B_um = float(surface["grid"]["a_um"]), float(surface["grid"]["b_um"])
    du, dv = np.pi / len(gu), np.pi / len(gv)

    def build_view(view: str):
        # Filled, depth-sorted quads (painter's algorithm). Anterior (x = +a) on the LEFT.
        polys, cells, depth = [], [], []
        for i in range(len(gu)):
            us = np.array([gu[i] - du / 2, gu[i] + du / 2, gu[i] + du / 2, gu[i] - du / 2])
            for j in range(len(gv)):
                if view == "dorsal" and np.cos(gv[j]) <= 0:
                    continue                     # the ventral half is hidden from above
                vs = np.array([gv[j] - dv / 2, gv[j] - dv / 2, gv[j] + dv / 2, gv[j] + dv / 2])
                X = A_um * np.cos(us)
                Y = B_um * np.sin(us) * np.cos(vs)
                Z = B_um * np.sin(us) * np.sin(vs)
                if view == "dorsal":
                    for sign in (1.0, -1.0):     # simulated half and its mirror image
                        polys.append(np.column_stack([-X, sign * Z])); cells.append((i, j)); depth.append(float(Y.mean()))
                else:
                    polys.append(np.column_stack([-X, Y])); cells.append((i, j)); depth.append(float(Z.mean()))
        order = np.argsort(depth)
        return [polys[k] for k in order], [cells[k] for k in order]

    view_geometry = {"dorsal": build_view("dorsal"), "lateral": build_view("lateral")}
    magma = plt.get_cmap("magma")
    cs = np.asarray(cross["grid"]["s_um"], float)
    cross_times_all = [float(v) for v in cross["times"]]
    show_times = [20.0, 30.0, 40.0, 60.0]
    cb = np.asarray([cross["fields"]["BR"][cross_times_all.index(tm)] for tm in show_times], float)
    widths = [cross["readouts"]["by_time"][str(tm)]["widths_own_max"]["0.5"]["um"] for tm in show_times]
    flank = [float(np.interp(60.0, cs, profile)) for profile in cb]
    paper = bmp_embryo.PAPER_DATA["umulis2010_fig4F"]
    model_dm = {tm: cross["fields"]["BR"][cross_times_all.index(tm)][0] for tm in (15.0, 30.0, 45.0, 60.0)}
    checks = {row["id"]: row for row in validation["checks"]}
    passed = [row for row in validation["checks"] if row.get("pass")]
    failed = [row for row in validation["checks"] if row.get("pass") is False]

    def render(fig: plt.Figure, t: float):
        ax = base_axes(fig)
        if t < 5.8:
            a = scene_alpha(t, 0, 5.8)
            ax.add_patch(Ellipse((0.73, 0.48), 0.35, 0.58, facecolor=mix(BG, VIOLET, 0.18), edgecolor=VIOLET, lw=2.4, alpha=a))
            for r in np.linspace(0.05, 0.14, 5):
                ax.add_patch(Circle((0.73, 0.64), r, fill=False, edgecolor=TEAL, lw=2, alpha=a * (0.9 - r * 3)))
            label(ax, 0.07, 0.70, "BMP SHUTTLING", 14, TEAL, "bold", alpha=a)
            wrapped(ax, 0.07, 0.63, "BMP shuttling patterns the Drosophila embryo", width=32, size=34, weight="bold", alpha=a)
            wrapped(ax, 0.07, 0.34, "Umulis et al. (2010)\nDevelopmental Cell 18:260–274\ndoi:10.1016/j.devcel.2010.01.006", width=42, size=16, color=MUTED, alpha=a)
        if 5.2 <= t < 13.8:
            a = scene_alpha(t, 5.2, 13.8)
            header(ax, "Mechanism", "A dorsal source and a lateral shuttle", "Cross-section schematic • not concentration-scaled", alpha=a)
            center = (0.50, 0.43)
            ax.add_patch(Ellipse(center, 0.72, 0.42, facecolor=PANEL, edgecolor=GRID, lw=2.2, alpha=a))
            # Dorsal 40% source and lateral Sog zones.
            ax.add_patch(Arc(center, 0.72, 0.42, theta1=54, theta2=126, color=TEAL, lw=16, alpha=a * 0.8))
            ax.add_patch(Arc(center, 0.72, 0.42, theta1=126, theta2=162, color=VIOLET, lw=13, alpha=a * 0.8))
            ax.add_patch(Arc(center, 0.72, 0.42, theta1=18, theta2=54, color=VIOLET, lw=13, alpha=a * 0.8))
            label(ax, 0.50, 0.69, "dorsal 40%: Dpp + Tsg", 15, TEAL, "bold", ha="center", alpha=a)
            label(ax, 0.18, 0.49, "Sog", 15, VIOLET, "bold", ha="center", alpha=a)
            label(ax, 0.82, 0.49, "Sog", 15, VIOLET, "bold", ha="center", alpha=a)
            label(ax, 0.50, 0.27, "Tld cleavage releases BMP near the dorsal midline", 15, MUTED, ha="center", alpha=a)
            travel = ramp((t - 5.2) % 2.2, 0.1, 2.0)
            for side in (-1, 1):
                xx = 0.50 + side * (0.28 * (1 - travel))
                yy = 0.48 + 0.15 * travel
                ax.add_patch(Circle((xx, yy), 0.016, facecolor=TEAL, edgecolor=TEXT, lw=0.8, alpha=a))
                ax.add_patch(Circle((xx + side * 0.017, yy - 0.012), 0.010, facecolor=VIOLET, edgecolor="none", alpha=a))
            label(ax, 0.50, 0.57, "Tld", 14, AMBER, "bold", ha="center", alpha=a * ramp(t, 7.0, 8.0))
            label(ax, 0.50, 0.53, "✦", 22, AMBER, "bold", ha="center", alpha=a * ramp(t, 7.2, 8.2))
        if 13.2 <= t < 25.8:
            a = scene_alpha(t, 13.2, 25.8)
            sim_min = 60.0 * ramp(t, 14.0, 24.2)
            k1 = min(int(np.searchsorted(st, sim_min, side="right")), len(st) - 1)
            k0 = max(0, k1 - 1)
            frac = 0.0 if st[k1] == st[k0] else (sim_min - st[k0]) / (st[k1] - st[k0])
            field = (1-frac) * brs[k0] + frac * brs[k1]
            header(ax, "Real SBP simulation", "Receptor-bound BMP concentrates dorsally", f"Orthographic projections • {grid_label} • colour = BR (nM)", alpha=a)
            for idx, (view, title) in enumerate((("dorsal", "Dorsal view"), ("lateral", "Lateral view (dorsal up)"))):
                polys, cells = view_geometry[view]
                values = np.array([field[i][j] for i, j in cells]) / vmax
                colors = magma(np.clip(values, 0.0, 1.0))
                colors[:, 3] = a
                p = fig.add_axes([0.06 + idx * 0.46, 0.22, 0.40, 0.49], facecolor=BG)
                p.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=colors, linewidths=0.35))
                p.add_patch(Ellipse((0, 0), 2 * A_um, 2 * B_um, fill=False, edgecolor=TEAL, lw=1.0, alpha=0.5 * a))
                p.set_xlim(-1.08 * A_um, 1.08 * A_um); p.set_ylim(-1.25 * B_um, 1.25 * B_um)
                p.set_aspect("equal"); p.axis("off"); p.set_title(title, color=TEXT, fontsize=15, pad=7, alpha=a)
                p.text(-1.02 * A_um, 0, "A", color=MUTED, fontsize=12, ha="right", va="center", alpha=a)
                p.text(1.02 * A_um, 0, "P", color=MUTED, fontsize=12, ha="left", va="center", alpha=a)
            key = fig.add_axes([0.935, 0.26, 0.012, 0.40])
            key.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", cmap="magma", alpha=a)
            key.set_xticks([]); key.set_yticks([0, 255]); key.set_yticklabels([f"{vmax:.0f}", "0"], color=MUTED, fontsize=10)
            key.yaxis.tick_right()
            for spine in key.spines.values(): spine.set_visible(False)
            label(ax, 0.945, 0.69, "nM", 11, MUTED, ha="center", alpha=a)
            label(ax, 0.50, 0.145, f"{sim_min:04.1f} min", 24, TEAL, "bold", ha="center", alpha=a)
            ax.plot([0.22, 0.78], [0.105, 0.105], color=GRID, lw=6, solid_capstyle="round", alpha=a)
            ax.plot([0.22, 0.22 + 0.56 * sim_min / 60], [0.105, 0.105], color=TEAL, lw=6, solid_capstyle="round", alpha=a)
            label(ax, 0.22, 0.075, "0", 11, MUTED, ha="center", alpha=a)
            label(ax, 0.78, 0.075, "60 min", 11, MUTED, ha="center", alpha=a)
        if 25.2 <= t < 34.8:
            a = scene_alpha(t, 25.2, 34.8)
            header(ax, "Cross-section at x/L = 0.5", "The signal contracts toward the dorsal midline",
                   "Solver output • dots: published Fig. 4F midline values (Umulis 2010)", alpha=a)
            p = fig.add_axes([0.10, 0.20, 0.60, 0.54], facecolor=PANEL)
            colors = ["#38bdf8", TEAL, VIOLET, PINK]
            reveal = ramp(t, 25.9, 31.5)
            for i, tm in enumerate(show_times):
                aa = a * ramp(reveal, i / 5, (i + 1) / 5)
                p.plot(cs, cb[i], color=colors[i], lw=2.6, alpha=aa, label=f"{tm:.0f} min")
            pa = a * ramp(t, 30.5, 31.5)
            p.scatter([0.0] * len(paper["times_min"]), paper["dm_BR_nM"], s=46, color=AMBER, zorder=5,
                      alpha=pa, label="paper (DM)")
            p.set_xlim(cs[0], 160.0); p.set_ylim(0, max(float(np.max(cb)), max(paper["dm_BR_nM"])) * 1.12)
            p.set_xlabel("distance from dorsal midline (µm)", color=MUTED, fontsize=11)
            p.set_ylabel("BR (nM)", color=MUTED, fontsize=11)
            p.grid(color=GRID, lw=0.7); p.tick_params(colors=MUTED, labelsize=10)
            for spine in p.spines.values(): spine.set_color(GRID)
            leg = p.legend(frameon=False, fontsize=10, loc="upper right")
            for tx in leg.get_texts(): tx.set_color(TEXT)
            rounded(ax, 0.74, 0.20, 0.21, 0.54, face=PANEL, edge=TEAL, alpha=a)
            label(ax, 0.845, 0.695, "measured", 14, TEAL, "bold", ha="center", alpha=a)
            label(ax, 0.755, 0.645, "time   FWHM    BR @60 µm", 11, MUTED, alpha=a)
            for i, tm in enumerate(show_times):
                label(ax, 0.755, 0.595 - i * 0.06, f"{tm:>2.0f} min  {widths[i]:5.0f} µm   {flank[i]:5.1f} nM", 11, TEXT, alpha=a)
            dm_err = [abs(model_dm[tm] - pv) / pv for tm, pv in zip((15.0, 30.0, 45.0, 60.0), paper["dm_BR_nM"])]
            wrapped(ax, 0.755, 0.33, f"60-min midline: model {model_dm[60.0]:.1f} vs paper {paper['dm_BR_nM'][3]:.1f} nM "
                    f"({100 * dm_err[3]:.0f}% low). Early times rise later than the paper.", width=30, size=10.5,
                    color=AMBER, alpha=a)
        if 34.2 <= t < 42.8:
            a = scene_alpha(t, 34.2, 42.8)
            header(ax, "Mutants at 60 min", "wild type · sog+/− · sog−/− · tld−/−", "Solver output, 57-node cross-section", alpha=a)
            p = fig.add_axes([0.10, 0.20, 0.80, 0.54], facecolor=PANEL)
            datasets = [("wild type", cb[-1], TEAL), ("sog+/−", np.asarray(sog["fields"]["BR"][-1]), VIOLET),
                        ("sog−/−", np.asarray(sog_null["fields"]["BR"][-1]), AMBER),
                        ("tld−/−", np.asarray(tld["fields"]["BR"][-1]), PINK)]
            for i, (name, values, color) in enumerate(datasets):
                pa = a * ramp(t, 35.0 + i * 0.7, 36.1 + i * 0.7)
                p.plot(cs, values, color=color, lw=3, label=f"{name}   midline {values[0]:.1f} nM", alpha=pa)
            p.set_xlim(cs[0], cs[-1]); p.set_ylim(bottom=0)
            p.set_xlabel("distance from dorsal midline (µm)", color=MUTED, fontsize=11)
            p.set_ylabel("BR (nM)", color=MUTED, fontsize=11)
            p.grid(color=GRID, lw=0.7); p.tick_params(colors=MUTED, labelsize=10)
            for spine in p.spines.values(): spine.set_color(GRID)
            leg = p.legend(frameon=False, fontsize=11, loc="upper right")
            for tx in leg.get_texts(): tx.set_color(TEXT)
            v4 = checks.get("V4", {}).get("measured", {})
            label(ax, 0.50, 0.12, f"tld−/− peak / wild-type peak = {v4.get('peak_ratio', 0):.1e}", 16, PINK, "bold", ha="center", alpha=a)
        if t >= 42.2:
            a = scene_alpha(t, 42.2, 48.01, fade=0.45)
            header(ax, "Validation against the papers", f"{validation['passed']} of {validation['total']} checks pass",
                   "Every check, pass or fail, is listed in the app with its measured numbers", alpha=a)
            left = [r for r in passed if r["id"] in ("S1", "V1", "V2", "V3", "V4", "V6", "V7")][:6]
            for i, row in enumerate(left):
                yy = 0.66 - i * 0.075
                label(ax, 0.08, yy, f"✓ {row['id']}", 14, GREEN, "bold", alpha=a)
                wrapped(ax, 0.145, yy + 0.012, row["claim"], width=48, size=11.5, alpha=a)
            for i, row in enumerate(failed[:4]):
                yy = 0.66 - i * 0.11
                label(ax, 0.56, yy, f"✗ {row['id']}", 14, AMBER, "bold", alpha=a)
                wrapped(ax, 0.625, yy + 0.012, row["claim"], width=40, size=11.5, color=MUTED, alpha=a)
            label(ax, 0.50, 0.115, "Unpublished inputs (FISH Sog field, image-based initial state) are the documented cause of the gaps.",
                  13, AMBER, "bold", ha="center", alpha=a)
    return render


def graph_blueprint() -> dict[str, Any]:
    """A transparent four-species system with one intentionally weak influence."""
    return {
        "type": "ODE",
        "nodes": [
            {"id": "EGFR", "name": "EGFR", "initial_value": 1.0},
            {"id": "RAS", "name": "RAS", "initial_value": 0.4},
            {"id": "ERK", "name": "ERK", "initial_value": 0.3},
            {"id": "DUSP", "name": "DUSP", "initial_value": 0.1},
        ],
        "parameters": {},
        "odes": {
            "EGFR": "-0.1*EGFR",
            "RAS": "1.0*EGFR - 0.2*RAS",
            "ERK": "1.0*RAS - 0.8*DUSP + 0.02*EGFR + 0.4*ERK/(1+ERK) - 0.2*ERK",
            "DUSP": "0.6*ERK - 0.2*DUSP",
        },
        "simulation_config": {"t_max": 20.0},
    }


def load_graph_data() -> dict[str, Any]:
    bp = graph_blueprint()
    model = ODEModel(bp)
    readable = derive_edges_from_odes(bp) or []
    # Exact strengths for this displayed system at its displayed initial state,
    # after removing pure first-order target turnover exactly as the app does.
    strengths = {
        ("EGFR", "RAS"): 1.0,
        ("RAS", "ERK"): 1.0,
        ("DUSP", "ERK"): -0.8,
        ("ERK", "ERK"): 0.4 / (1.0 + 0.3) ** 2,
        ("EGFR", "ERK"): 0.02,
        ("ERK", "DUSP"): 0.6,
    }
    if any(e["source"] == "EGFR" and e["target"] == "ERK" for e in readable):
        raise RuntimeError("The intended <5% weak influence was not folded")
    return {"bp": bp, "model": model, "readable": readable, "strengths": strengths}


def graph_renderer(data: dict[str, Any]) -> Callable[[plt.Figure, float], None]:
    bp, strengths = data["bp"], data["strengths"]
    pos = {"EGFR": (0.20, 0.47), "RAS": (0.42, 0.63), "ERK": (0.68, 0.52), "DUSP": (0.52, 0.29)}

    def network(ax, alpha=1.0, progress=1.0, show_weak=False, labels=False):
        edges = [
            ("EGFR", "RAS", GREEN, False, 0.0),
            ("RAS", "ERK", GREEN, False, 0.0),
            ("ERK", "DUSP", GREEN, False, 0.0),
            ("DUSP", "ERK", PINK, True, 0.0),
        ]
        for i, (s, d, c, inhib, curve) in enumerate(edges):
            ss, dd = np.array(pos[s]), np.array(pos[d]); vec = dd - ss; unit = vec / np.linalg.norm(vec)
            draw_arrow(ax, ss + unit * 0.058, dd - unit * 0.064, c, ramp(progress, i/6, (i+2)/6), inhib, alpha, 2.6, curve=curve)
        # Genuine ERK self-activation from +0.4*ERK/(1+ERK), drawn as a loop.
        pa = alpha * ramp(progress, 0.55, 0.90)
        ax.add_patch(Arc((0.73, 0.58), 0.15, 0.15, theta1=35, theta2=325, color=GREEN, lw=2.6, alpha=pa))
        draw_arrow(ax, (0.776, 0.626), (0.776, 0.625), GREEN, 1, False, pa, 2.2)
        if show_weak:
            ss, dd = np.array(pos["EGFR"]), np.array(pos["ERK"]); unit = (dd-ss)/np.linalg.norm(dd-ss)
            draw_arrow(ax, ss + unit*0.06, dd-unit*0.065, TEAL, 1, False, alpha*0.68, 1.5, dashed=True, curve=-0.12)
            if labels:
                label(ax, 0.43, 0.43, "weak 0.02", 12, TEAL, "bold", ha="center", alpha=alpha)
        for name, xy in pos.items(): node(ax, xy, name, TEAL if name in ("EGFR", "RAS") else VIOLET, alpha, 0.048)

    def render(fig: plt.Figure, t: float):
        ax = base_axes(fig)
        if t < 5.6:
            a = scene_alpha(t, 0, 5.6)
            label(ax, 0.50, 0.84, "INTERACTION GRAPH", 14, TEAL, "bold", ha="center", alpha=a)
            label(ax, 0.50, 0.755, "Reading the interaction graph", 34, TEXT, "bold", ha="center", alpha=a)
            network(ax, a, ramp(t, 0.8, 4.2), False)
            label(ax, 0.50, 0.15, "A readable view derived from the equations", 17, MUTED, ha="center", alpha=a)
        if 5.0 <= t < 12.5:
            a = scene_alpha(t, 5.0, 12.5)
            header(ax, "Definition", "An edge is an equation dependency", "The source appears in the target's rate of change", alpha=a)
            rounded(ax, 0.09, 0.51, 0.82, 0.17, face=PANEL, edge=TEAL, alpha=a)
            label(ax, 0.50, 0.595, "$d[ERK]/dt = 1.0[RAS] - 0.8[DUSP] + \\cdots$", 24, TEXT, ha="center", alpha=a)
            node(ax, (0.27, 0.32), "RAS", TEAL, a)
            node(ax, (0.73, 0.32), "ERK", VIOLET, a)
            draw_arrow(ax, (0.33, 0.32), (0.67, 0.32), GREEN, ramp(t, 6.7, 8.3), False, a)
            label(ax, 0.50, 0.22, "RAS appears in d[ERK]/dt", 16, MUTED, "bold", ha="center", alpha=a)
        if 11.9 <= t < 18.8:
            a = scene_alpha(t, 11.9, 18.8)
            header(ax, "Sign", "The Jacobian determines arrow type", "Evaluated at the model's own initial state", alpha=a)
            rounded(ax, 0.08, 0.52, 0.40, 0.17, face=PANEL, edge=GREEN, alpha=a)
            label(ax, 0.28, 0.635, "ACTIVATION", 14, GREEN, "bold", ha="center", alpha=a)
            label(ax, 0.28, 0.57, "∂(dERK/dt)/∂RAS = +1.0", 18, TEXT, "bold", ha="center", alpha=a)
            rounded(ax, 0.52, 0.52, 0.40, 0.17, face=PANEL, edge=PINK, alpha=a)
            label(ax, 0.72, 0.635, "INHIBITION", 14, PINK, "bold", ha="center", alpha=a)
            label(ax, 0.72, 0.57, "∂(dERK/dt)/∂DUSP = −0.8", 18, TEXT, "bold", ha="center", alpha=a)
            node(ax, (0.20, 0.30), "RAS", TEAL, a); node(ax, (0.48, 0.30), "ERK", VIOLET, a); node(ax, (0.80, 0.30), "DUSP", VIOLET, a)
            draw_arrow(ax, (0.26, 0.30), (0.42, 0.30), GREEN, ramp(t, 13.3, 14.5), False, a)
            draw_arrow(ax, (0.74, 0.30), (0.54, 0.30), PINK, ramp(t, 14.0, 15.2), True, a)
        if 18.2 <= t < 25.8:
            a = scene_alpha(t, 18.2, 25.8)
            header(ax, "Self edges", "Autoregulation is not turnover", "Pure −k·X loss is removed before edge derivation", alpha=a)
            rounded(ax, 0.08, 0.22, 0.41, 0.48, face=PANEL, edge=GREEN, alpha=a)
            node(ax, (0.285, 0.46), "ERK", VIOLET, a)
            ax.add_patch(Arc((0.345, 0.52), 0.19, 0.19, theta1=30, theta2=325, color=GREEN, lw=3, alpha=a))
            draw_arrow(ax, (0.40, 0.57), (0.40, 0.568), GREEN, 1, False, a, 2.5)
            label(ax, 0.285, 0.65, "+0.4·ERK/(1+ERK)", 16, GREEN, "bold", ha="center", alpha=a)
            label(ax, 0.285, 0.30, "positive self-regulation", 14, MUTED, ha="center", alpha=a)
            rounded(ax, 0.53, 0.22, 0.39, 0.48, face=PANEL, edge=GRID, alpha=a)
            node(ax, (0.725, 0.46), "EGFR", TEAL, a)
            label(ax, 0.725, 0.65, "−0.1·EGFR", 16, TEXT, "bold", ha="center", alpha=a)
            label(ax, 0.725, 0.30, "first-order turnover\n(no self-edge)", 14, MUTED, "bold", ha="center", alpha=a, linespacing=1.3)
        if 25.2 <= t < 33.8:
            a = scene_alpha(t, 25.2, 33.8)
            header(ax, "Readable view", "Weak influences are folded, not deleted", "Cutoff: <5% of the strongest influence on each target", alpha=a)
            network(ax, a, 1, False)
            rounded(ax, 0.77, 0.23, 0.18, 0.42, face=PANEL, edge=GRID, alpha=a)
            label(ax, 0.86, 0.60, "ERK inputs", 14, TEXT, "bold", ha="center", alpha=a)
            rows = [("RAS", "+1.00", GREEN), ("DUSP", "−0.80", PINK), ("ERK", "+0.237", GREEN), ("EGFR", "+0.02", MUTED)]
            for i, (name, value, color) in enumerate(rows):
                yy = 0.53 - i * 0.065
                label(ax, 0.79, yy, name, 12, color, "bold", alpha=a if i < 3 else a*0.48)
                label(ax, 0.93, yy, value, 12, color, ha="right", alpha=a if i < 3 else a*0.48)
            label(ax, 0.86, 0.27, "0.02 < 0.05 × 1.00", 11, AMBER, "bold", ha="center", alpha=a)
        if 33.2 <= t < 37.8:
            a = scene_alpha(t, 33.2, 37.8)
            header(ax, "Show all connections", "Reveal folded edges on demand", "Minor influences return as thin dashed connections", alpha=a)
            rounded(ax, 0.72, 0.70, 0.22, 0.075, face=PANEL_2, edge=TEAL, alpha=a)
            label(ax, 0.735, 0.738, "Show all connections", 11, TEXT, "bold", alpha=a)
            ax.add_patch(Circle((0.915, 0.738), 0.020, facecolor=TEAL, edgecolor="none", alpha=a))
            network(ax, a, 1, True, True)
        if t >= 37.2:
            a = scene_alpha(t, 37.2, 40.01, fade=0.4)
            label(ax, 0.50, 0.68, "The graph is a readable view.", 31, TEXT, "bold", ha="center", alpha=a)
            rounded(ax, 0.17, 0.33, 0.66, 0.20, face=PANEL, edge=TEAL, alpha=a)
            label(ax, 0.50, 0.455, "The Equations tab is always complete.", 24, TEAL, "bold", ha="center", alpha=a)
            label(ax, 0.50, 0.385, "Every term remains visible — including folded connections and turnover.", 15, MUTED, ha="center", alpha=a)
            label(ax, 0.50, 0.20, "Structure from equations • signs from partial derivatives • clarity by thresholding", 14, TEXT, ha="center", alpha=a)
    return render


def probe(path: Path) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "format=duration,size:stream=codec_name,width,height,r_frame_rate,pix_fmt",
           "-of", "json", str(path)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def write_manifest():
    payload = []
    for key in ("pipeline", "bmp", "graph"):
        row = dict(VIDEO_META[key])
        path = OUT / row["file"]
        if path.exists():
            info = probe(path)
            row["duration_s"] = round(float(info["format"]["duration"]), 3)
        payload.append(row)
    (OUT / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("pipeline", "bmp", "graph"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [args.only] if args.only else ["pipeline", "bmp", "graph"]
    loaders = {"pipeline": load_pipeline_data, "bmp": load_bmp_data, "graph": load_graph_data}
    renderers = {"pipeline": pipeline_renderer, "bmp": bmp_renderer, "graph": graph_renderer}
    for key in keys:
        print(f"Preparing real outputs for {key}...", flush=True)
        data = loaders[key]()
        print(f"Rendering {VIDEO_META[key]['file']}...", flush=True)
        encode_video(key, renderers[key](data))
        info = probe(OUT / VIDEO_META[key]["file"])
        stream = info["streams"][0]
        print(f"  wrote {info['format']['size']} bytes, {float(info['format']['duration']):.3f} s, {stream['codec_name']} {stream['width']}x{stream['height']} {stream['pix_fmt']}", flush=True)
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
