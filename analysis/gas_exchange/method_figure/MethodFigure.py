"""Methods figure for the gas-exchange protocol.

Everything is drawn from one production trajectory (873 K, cell 3) and the
check at 99.4 ps, which removed one CO and added one O2.
(a) OVITO render of the cell at that check, with the removed CO and the added
    O2 marked. Same rendering as the Snapshots figure of the Discussion.
(b) The steps of the controller.
(c) The controller log of that cell over 140 ps.

Three steps, the first two are skipped when their output exists:
  1. prepare  writes MethodFrame.xyz and MethodFrame.json from the arrays of
              ../BuildCache.py and the results of ../GasExchangeAnalysis.py
              (873_3.npz and analysis_873_3.pkl)
  2. render   runs MethodRender.py with a Python that has the ovito package
              (environment variable OVITO_PYTHON) -> Panel_Method.png
  3. compose  writes Fig_Method.pdf and Fig_Method.png

Usage:
    python MethodFigure.py            # compose, prepare and render if needed
    python MethodFigure.py --redo     # all three steps
"""

import argparse
import ast
import os
import json
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import Paths as RP

TEMPERATURE, CELL, CHECK = 873, 3, 1041
OVITO_PYTHON = os.environ.get("OVITO_PYTHON", sys.executable)  # a Python that has the ovito package
SYMBOL = {1: "O", 2: "Zr", 3: "C"}
MARKER_PAD = (1.8, 1.8)  # A, distance of the two scale markers from the cell in x and z

# Greys throughout, with the red of the oxygen atoms as the only accent (oxygen added).
INK, MUTED, RULE, FILL = "0.10", "0.42", "0.55", "0.87"
OXYGEN = "#b3241c"
ZR_COLOUR, C_COLOUR, O_COLOUR = (0.72, 0.75, 0.80), (0.15, 0.15, 0.15), (0.82, 0.12, 0.10)
SIZE = 8.5

plt.rcParams.update({"font.family": "Times New Roman", "font.size": SIZE, "mathtext.fontset": "stix",
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "pdf.fonttype": 42})


# ----------------------------------------------------------------- prepare

def circular_distance(a, b, period):
    return np.abs((a - b + period / 2) % period - period / 2)


def prepare():
    from BuildCache import DEFAULT_CACHE
    from GasExchangeAnalysis import distance_matrix, free_groups

    with open(Path(DEFAULT_CACHE) / f"analysis_{TEMPERATURE}_{CELL}.pkl", "rb") as handle:
        res = pickle.load(handle)
    with np.load(Path(DEFAULT_CACHE) / f"{TEMPERATURE}_{CELL}.npz") as npz:
        starts = res["starts"]
        before = int(starts[starts > 0][CHECK - 1] - 1)
        frames = []
        for index in (before, before + 1):
            n = int(npz["n"][index])
            frames.append({"el": npz["el"][index, :n], "pos": npz["pos"][index, :n].astype(float),
                           "lat": npz["lat"][index]})
    time_ps = float(res["time"][before]) / 1000.0

    def molecules(frame):
        dist = distance_matrix(frame["pos"], frame["lat"], np.linalg.inv(frame["lat"]))
        return [g for g in free_groups(frame["el"], dist, 1.0) if 2 <= len(g) <= 3]

    first, second = frames
    product = [g for g in molecules(first) if 3 in first["el"][list(g)]][0]
    added = [g for g in molecules(second) if set(second["el"][list(g)]) == {1}][0]
    size = np.array([first["lat"][0, 0], first["lat"][1, 1], first["lat"][2, 2]])

    # Along x: cut the slab between two Zr planes so that the gas is in the middle.
    zr = np.sort(first["pos"][first["el"] == 2, 0] % size[0])
    gaps = np.diff(np.append(zr, zr[0] + size[0]))
    widest = int(np.argmax(gaps))
    gas_length = float(gaps[widest])
    gas_centre = (zr[widest] + gas_length / 2) % size[0]
    between_planes = [(zr[i] + gaps[i] / 2) % size[0] for i in range(len(zr)) if 1.0 < gaps[i] < 5.0]
    target = (gas_centre + size[0] / 2) % size[0]
    cut = min(between_planes, key=lambda x: circular_distance(x, target, size[0]))

    def centre_of(frame, group):
        ref = frame["pos"][group[0]]
        delta = frame["pos"][list(group)] - ref
        delta -= size * np.round(delta / size)
        return ref + delta.mean(axis=0), ref + delta

    co_centre, co_atoms = centre_of(first, product)
    o2_centre, o2_atoms = centre_of(second, added)

    # Along z (vertical in the picture): keep both molecules away from the edges.
    shifts = np.linspace(0, size[2], 200, endpoint=False)
    clearance = [min(min((c[2] + s) % size[2], size[2] - (c[2] + s) % size[2]) for c in (co_centre, o2_centre))
                 for s in shifts]
    z_shift = float(shifts[int(np.argmax(clearance))])

    def place(points, anchor=None):
        """Shift into the picture frame. A molecule moves as a whole with its centre."""
        offset = np.array([-cut, 0.0, z_shift])
        if anchor is None:
            return (points + offset) % size
        centre = (anchor + offset) % size
        return points - anchor + centre

    keep = np.array([i for i in range(len(first["el"])) if i not in product])
    species = [SYMBOL[int(e)] for e in first["el"][keep]]
    position = place(first["pos"][keep])
    co_position = place(co_atoms, co_centre)
    o2_position = place(o2_atoms, o2_centre)
    species += [SYMBOL[int(e)] for e in first["el"][list(product)]] + ["O", "O"]
    position = np.vstack([position, co_position, o2_position])
    markers = np.array([[-MARKER_PAD[0], size[1] / 2, -MARKER_PAD[1]],
                        [size[0] + MARKER_PAD[0], size[1] / 2, size[2] + MARKER_PAD[1]]])
    species += ["Mk", "Mk"]
    position = np.vstack([position, markers])

    with open(HERE / "MethodFrame.xyz", "w", encoding="utf-8") as handle:
        handle.write(f"{len(species)}\n")
        handle.write(f'Lattice="{size[0]:.5f} 0 0 0 {size[1]:.5f} 0 0 0 {size[2]:.5f}" '
                     'Properties=species:S:1:pos:R:3 pbc="F F F"\n')
        for name, p in zip(species, position):
            handle.write(f"{name} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")

    left_edge = float((zr[widest] - cut) % size[0])
    info = {"trajectory": f"{TEMPERATURE}_{CELL}", "check": CHECK, "time_ps": time_ps, "size": size.tolist(),
            "markers": markers[:, [0, 2]].tolist(), "gas_edges": [left_edge, left_edge + gas_length],
            "co": place(co_centre[None], co_centre)[0, [0, 2]].tolist(),
            "o2": place(o2_centre[None], o2_centre)[0, [0, 2]].tolist(),
            "product": "".join(sorted(SYMBOL[int(e)] for e in first["el"][list(product)]))}
    (HERE / "MethodFrame.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("prepared", info)


# ----------------------------------------------------------------- compose

def load_panel(info):
    """The render, with the scale markers removed, and its extent in A (x, z)."""
    image = mpimg.imread(HERE / "Panel_Method.png")[:, :, :3].copy()
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    marker = (r > 0.15) & (np.abs(r - b) < 0.25 * r) & (g < 0.45 * r)
    labels, count = ndimage.label(marker)
    sizes = ndimage.sum(marker, labels, range(1, count + 1))
    biggest = np.argsort(sizes)[-2:] + 1
    centres = sorted((ndimage.center_of_mass(marker, labels, k) for k in biggest), key=lambda c: -c[0])
    image[ndimage.binary_dilation(marker, iterations=8)] = 1.0
    (row_a, col_a), (row_b, col_b) = centres  # a is the marker at low z, the lower one in the picture
    (xa, za), (xb, zb) = info["markers"]
    per_col, per_row = (xb - xa) / (col_b - col_a), (zb - za) / (row_b - row_a)
    height, width = image.shape[:2]
    x_of = lambda col: xa + (col - col_a) * per_col
    z_of = lambda row: za + (row - row_a) * per_row
    extent = (x_of(-0.5), x_of(width - 0.5), z_of(height - 0.5), z_of(-0.5))
    return image, extent


WIDTH = 131.2 / 25.4

TAG = 10

def element_handles():
    return [Line2D([0], [0], marker="o", ls="none", markersize=8, markerfacecolor=ZR_COLOUR,
                   markeredgecolor="0.45", label="Zr"),
            Line2D([0], [0], marker="o", ls="none", markersize=5, markerfacecolor=C_COLOUR,
                   markeredgecolor="none", label="C"),
            Line2D([0], [0], marker="o", ls="none", markersize=4.5, markerfacecolor=O_COLOUR,
                   markeredgecolor="none", label="O")]


def callout(ax, label, ring_edge, text_at, colour, style, align="center"):
    ax.annotate(label, xy=ring_edge, xytext=text_at, ha=align, va="center", fontsize=SIZE, color=INK, zorder=5,
                linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.25", facecolor="white", edgecolor=colour,
                          linewidth=0.8),
                arrowprops=dict(arrowstyle=style, color=colour, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=8))


V_LEFT, V_RIGHT = 3.6, 8.6

def cell_vertical(ax, info):
    """The cell with its long axis upright. Picture coordinates are (z, x)."""
    image, extent = load_panel(info)
    length, _, height = info["size"]
    ax.imshow(np.transpose(image, (1, 0, 2)), extent=(extent[3], extent[2], extent[1], extent[0]),
              interpolation="antialiased", zorder=1)
    ax.add_patch(Rectangle((0, 0), height, length, facecolor="none", edgecolor="0.30", linewidth=0.6, zorder=3))
    for centre, colour, label, style in ((info["co"], INK, "CO, CO$_2$\nremoved", "<|-"),
                                         (info["o2"], OXYGEN, "O$_2$\nadded", "-|>")):
        point = (centre[1], centre[0])
        ax.add_patch(Circle(point, 1.3, facecolor="none", edgecolor=colour, linewidth=0.9, zorder=4))
        callout(ax, label, (point[0] + 1.3, point[1]), (height + 4.6, point[1]), colour, style)
    low, high = info["gas_edges"]
    beside, label_x = -1.5, -2.6
    for edge in (low, high):
        ax.plot([beside - 0.3, 0.0], [edge, edge], color=RULE, lw=0.5, zorder=2)
    ax.annotate("", xy=(beside, low), xytext=(beside, high),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(label_x, (low + high) / 2, "gas volume", ha="center", va="center", rotation=90, fontsize=SIZE,
            color=INK)
    for y in (low / 2, (high + length) / 2):
        ax.text(label_x, y, "oxidised ZrC", ha="center", va="center", rotation=90, fontsize=SIZE, color=MUTED)
    ax.set_xlim(-V_LEFT, height + V_RIGHT)
    ax.set_ylim(-0.6, length + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")


LIST_STEPS = ["Find isolated groups of two or three atoms",
              "Delete CO and CO$_2$, keep O$_2$",
              "Count O$_2$ in the gas, $N$, and smooth it, $\\tilde{N}$",
              "Measure the gas volume, set $N_\\mathrm{tol} = 0.5\\,f_V$",
              "Add one O$_2$ if $\\tilde{N} \\leq N_\\mathrm{tol}$ and $N < N_\\mathrm{tol}$"]

def badge(ax, x, y, number):
    ax.add_patch(Circle((x, y), 0.075, facecolor="0.25", edgecolor="white", linewidth=0.8, zorder=3))
    ax.text(x, y - 0.005, str(number), ha="center", va="center", fontsize=7, color="white", fontweight="bold",
            zorder=4)


def steps_list(ax, width_in, height_in, steps):
    """Numbered steps on a vertical line that closes into a loop."""
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.axis("off")
    line = SIZE * 1.2 / 72.0
    rows = [text.count("\n") + 1 for text in steps]
    top, bottom = height_in - 0.10, 0.30
    between = (top - bottom - sum(rows) * line) / (len(steps) - 1)
    x_badge, x_text, y, centres = 0.30, 0.50, top, []
    for number, (text, n) in enumerate(zip(steps, rows), start=1):
        centre = y - n * line / 2
        centres.append(centre)
        badge(ax, x_badge, centre, number)
        ax.text(x_text, centre, text, ha="left", va="center", fontsize=SIZE, color=INK, linespacing=1.2)
        y -= n * line + between
    ax.plot([x_badge, x_badge], [centres[0], centres[-1]], color=INK, lw=0.7, zorder=1)
    x_loop, low = 0.08, 0.12
    ax.plot([x_badge, x_badge, x_loop, x_loop], [centres[-1], low, low, centres[0]], color=INK, lw=0.7, zorder=1,
            solid_joinstyle="miter")
    ax.add_patch(FancyArrowPatch((x_loop, centres[0]), (x_badge - 0.08, centres[0]), arrowstyle="-|>",
                                 mutation_scale=6, color=INK, lw=0.7, shrinkA=0, shrinkB=0, zorder=2))
    ax.text(x_text, low, "repeated every 80 MD steps", ha="left", va="center", fontsize=SIZE, color=MUTED)


def log_panel(fig, box, info, key_above=True, tick_step=20):
    """Event strip over the O2 counts. box = (left, bottom, width, height) in figure fractions."""
    left, bottom, width, height = box
    strip = 0.20 * height
    ax = fig.add_axes([left, bottom, width, height - strip - 0.02 * height])
    ax_events = fig.add_axes([left, bottom + height - strip, width, strip])
    log = pd.read_csv(RP.gas_log(TEMPERATURE, CELL))
    time = log["Time (fs)"].values / 1000.0
    added = np.diff(log["O2 Added"].values, prepend=log["O2 Added"].values[0]) > 0
    removed = log["Gas Removed"].apply(lambda s: len(ast.literal_eval(s))).values > 0
    threshold = 0.5 * log["Free Gas Fraction"].values
    ceiling = 2.3
    shown = (time > 0) & (time <= 140)
    check = info["time_ps"]

    rows = (("O$_2$ added", added, OXYGEN, 1.0), ("CO or CO$_2$ removed", removed, INK, 0.0))
    for _, mask, colour, row in rows:
        ax_events.axhline(row, color="0.88", lw=0.5, zorder=0)
        ax_events.vlines(time[mask & shown], row - 0.36, row + 0.36, color=colour, lw=1.7, zorder=2)
    ax_events.axvline(check, color=RULE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax_events.set_xlim(0, 140)
    ax_events.set_ylim(-0.55, 1.55)
    ax_events.set_yticks([])
    ax_events.tick_params(left=False, bottom=False, labelbottom=False)
    ax_events.spines[:].set_visible(False)
    keys = [Line2D([0], [0], marker="|", ls="none", markersize=9, markeredgewidth=2.2, color=colour, label=name)
            for name, _, colour, _ in rows]
    ax_events.legend(handles=keys, loc="lower right" if key_above else "lower left",
                     bbox_to_anchor=(1.0, 1.0) if key_above else (0.0, 1.0), ncols=2, fontsize=SIZE, frameon=False,
                     handlelength=0.7, handletextpad=0.35, columnspacing=1.5, borderaxespad=0.25)

    count = np.clip(log["O2 Count"].values.astype(float), 0, ceiling + 1)
    smoothed = np.where(log["Smoothed O2 Count"].values > ceiling + 0.05, np.nan, log["Smoothed O2 Count"].values)
    ax.fill_between(time, 0, count, step="post", color=FILL, lw=0, zorder=1)
    ax.plot(time, threshold, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=3)
    ax.plot(time, smoothed, color=INK, lw=1.2, zorder=4)
    ax.axvline(check, color=RULE, lw=0.5, ls=(0, (3, 2)), zorder=2)
    ax.text(check - 1.5, 1.24, "(a)", fontsize=SIZE, color=INK, ha="right", va="center", zorder=6)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, ceiling)
    ax.set_yticks([0, 1, 2])
    ax.set_xticks(np.arange(0, 141, tick_step))
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("O$_2$ in the gas")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)
    handles = [Patch(facecolor=FILL, edgecolor="none", label="count, $N$"),
               Line2D([0], [0], color=INK, lw=1.2, label="smoothed, $\\tilde{N}$"),
               Line2D([0], [0], color=INK, lw=0.9, ls=(0, (4, 2)), label="threshold, $N_\\mathrm{tol}$")]
    narrow = width * fig.get_figwidth() < 3.2
    legend = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.995), ncols=1 if narrow else 3,
                       fontsize=SIZE, frameon=True, handlelength=1.7, handletextpad=0.5, columnspacing=1.4,
                       borderaxespad=0.0, borderpad=0.25, labelspacing=0.25)
    legend.get_frame().set(facecolor="white", edgecolor="none", alpha=1.0)
    legend.set_zorder(5)


class Sheet:
    """A figure addressed in inches from the top left."""

    def __init__(self, height):
        self.height = height
        self.fig = plt.figure(figsize=(WIDTH, height))

    def box(self, left, top, width, height):
        return [left / WIDTH, 1 - (top + height) / self.height, width / WIDTH, height / self.height]

    def axes(self, left, top, width, height):
        return self.fig.add_axes(self.box(left, top, width, height))

    def tag(self, text, left, top):
        self.fig.text(left / WIDTH, 1 - (top + 0.035) / self.height, text, fontsize=TAG, fontweight="bold",
                      ha="left", va="top", color="black")

    def save(self, name):
        self.fig.savefig(HERE / f"{name}.png", dpi=300)
        self.fig.savefig(HERE / f"{name}.pdf", dpi=600)
        plt.close(self.fig)
        print("saved", name, round(WIDTH, 2), "x", round(self.height, 2), "in")


def layout_side(info):
    """The figure: upright cell on the left, list of steps and log on the right."""
    length, _, cell = info["size"]
    side, tag = 0.04, 0.27
    cell_height = 4.05
    scale = cell_height / (length + 1.2)
    cell_width = scale * (cell + V_LEFT + V_RIGHT)
    list_height, c = 1.42, 2.02
    total = tag + cell_height + 0.30
    sheet = Sheet(total)
    sheet.tag("(a) One check of the cell", side, 0.0)
    ax = sheet.axes(side, tag, cell_width, cell_height)
    cell_vertical(ax, info)
    ax.legend(handles=element_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.005), ncols=3, fontsize=SIZE,
              frameon=False, handletextpad=0.25, columnspacing=1.1, borderaxespad=0.0)
    right = side + cell_width + 0.22
    right_width = WIDTH - right - 0.06
    sheet.tag("(b) Controller", right, 0.0)
    steps_list(sheet.axes(right, tag, right_width, list_height), right_width, list_height, LIST_STEPS)
    y = tag + list_height + 0.10
    sheet.tag("(c) Oxygen in the gas of this cell", right, y)
    log_panel(sheet.fig, sheet.box(right + 0.36, y + tag + 0.26, right_width - 0.36 - 0.08, c - 0.26), info,
              key_above=False, tick_step=20)
    sheet.save("Fig_Method")


def compose():
    info = json.loads((HERE / "MethodFrame.json").read_text(encoding="utf-8"))
    layout_side(info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="store_true", help="prepare and render again")
    args = parser.parse_args()
    if args.redo or not (HERE / "MethodFrame.json").exists():
        prepare()
    if args.redo or not (HERE / "Panel_Method.png").exists():
        subprocess.run([OVITO_PYTHON, str(HERE / "MethodRender.py")], check=True)
    compose()


if __name__ == "__main__":
    main()
