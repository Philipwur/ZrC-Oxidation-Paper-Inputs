"""
SnapshotsFigure.py
=======================================================================
Stage 2 of the Snapshots figure: compose
the two OVITO panels from SnapshotsRender.py into the publication
figure at print scale (131.2 mm width, Times New Roman, 10 pt).

Layout logic:
  - The crop window is computed from the panels' content, the union of
    dense columns across BOTH panels with equal side margins, so the
    slab sits centred and both temperatures share one magnification.
    Sparse far-field gas atoms do not widen the window.
  - The "chains" arrows target the centroid of the largest connected
    amber-pixel cluster detected in each panel, so they can only point
    at rendered chain carbon.
  - imshow uses interpolation='none' and the PNG is written at 600 dpi
    to keep the ray-traced detail crisp in print.

Run with the base environment:  python SnapshotsFigure.py
Output: SnapshotsResults/Snapshots.png/.pdf
The paper uses the PDF as Snapshots.pdf.
"""

import os

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

ScriptDir = os.path.dirname(os.path.abspath(__file__))
ResultsDir = os.path.join(ScriptDir, "SnapshotsResults")

Panels = [("973_3", "(a) 973 K"), ("1273_3", "(b) 1273 K")]

DenseColMin = 25         # px of non-white per column to count as content
MarginPx = 60            # equal side margins around the union window

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10,
    "mathtext.fontset": "stix",
})

ZrCol = (0.72, 0.75, 0.80)
CCol = (0.15, 0.15, 0.15)
ChainCol = (0.90, 0.60, 0.05)
OCol = (0.82, 0.12, 0.10)

Halo = [pe.withStroke(linewidth=2.2, foreground="white")]

# Annotations: chains arrow (auto-targeted at amber) and delamination
# label only on the 1273 K panel; 973 K carries no in-panel labels
# (its chains are sparse by design and the caption carries the point).
ChainText = {"1273_3": (0.72, 0.90)}
Delam = {}   # no delamination marker


def NonWhite(Img):
    return (Img[:, :, :3] < 0.985).any(axis=2)


def AmberMask(Img):
    R, G, B = Img[:, :, 0], Img[:, :, 1], Img[:, :, 2]
    return (R > 0.42) & (B < 0.55 * G) & (G > 0.42 * R) & (G < 0.88 * R)


def LargestClusterCentroid(Mask):
    LabelArr, N = ndimage.label(Mask, structure=np.ones((3, 3)))
    if N == 0:
        return None
    Sizes = ndimage.sum(Mask, LabelArr, range(1, N + 1))
    Best = int(np.argmax(Sizes)) + 1
    Cy, Cx = ndimage.center_of_mass(Mask, LabelArr, Best)
    return Cx, Cy, float(Sizes[Best - 1])


def Main():
    Imgs = {T: mpimg.imread(os.path.join(ResultsDir, f"Panel_{T}.png"))
            for T, _ in Panels}

    # Per-panel horizontal windows of one COMMON width (identical
    # magnification), each centred on its own content-mass centroid so
    # the slab sits centred in both panels. Isolated gas atoms shift
    # the centroid negligibly and do not widen the window.
    ColCounts = {T: NonWhite(I).sum(axis=0) for T, I in Imgs.items()}
    Spans = {}
    Centres = {}
    for T, Counts in ColCounts.items():
        Dense = np.where(Counts >= DenseColMin)[0]
        Spans[T] = Dense.max() - Dense.min()
        CountsF = Counts.astype(np.float64)
        Centres[T] = float((np.arange(len(CountsF)) * CountsF).sum()
                           / CountsF.sum())
    Width = max(Spans.values()) + 2 * MarginPx
    XWin = {}
    for T, I in Imgs.items():
        # Centre on the content centroid, then nudge (common width kept)
        # to the nearby offset whose window edges cut through the fewest
        # rendered pixels, so no atom is clipped at a panel boundary.
        NW = NonWhite(I)
        C = int(round(Centres[T]))
        Base = int(np.clip(C - Width // 2, 0, I.shape[1] - Width))
        Best, BestCost = Base, None
        for Off in range(-160, 161, 8):
            X0 = int(np.clip(Base + Off, 0, I.shape[1] - Width))
            Cost = int(NW[:, X0:X0 + 6].sum() +
                       NW[:, X0 + Width - 6:X0 + Width].sum())
            if BestCost is None or Cost < BestCost or \
               (Cost == BestCost and abs(X0 - Base) < abs(Best - Base)):
                Best, BestCost = X0, Cost
        XWin[T] = (Best, Best + Width)
    RowCounts = [NonWhite(I).sum(axis=1) for I in Imgs.values()]
    DenseRows = np.where(np.maximum(*RowCounts) >= 10)[0]
    Y0 = max(0, DenseRows.min() - 25)
    Y1 = min(Imgs[Panels[0][0]].shape[0], DenseRows.max() + 25)
    print(f"common width {Width}, windows {XWin}, y [{Y0}, {Y1}]")

    FigW = 131.2 / 25.4
    PanelAspect = (Y1 - Y0) / Width
    PanelWIn = 0.5 * (FigW - 0.10)
    TagRoomIn = 0.22
    FigH = PanelWIn * PanelAspect + 0.42 + TagRoomIn
    Fig, Axes = plt.subplots(1, 2, figsize=(FigW, FigH))
    Fig.subplots_adjust(left=0.002, right=0.998,
                        top=1.0 - TagRoomIn / FigH,
                        bottom=0.42 / FigH, wspace=0.015)

    for Ax, (TrajId, Tag) in zip(Axes, Panels):
        Crop = Imgs[TrajId][Y0:Y1, XWin[TrajId][0]:XWin[TrajId][1]]
        Ax.imshow(Crop, interpolation="none")
        Ax.set_xticks([])
        Ax.set_yticks([])
        Ax.axis("off")
        Ax.text(0.01, 1.03, Tag, transform=Ax.transAxes, ha="left",
                va="bottom", fontsize=10, fontweight="bold")
        if TrajId in ChainText:
            Found = LargestClusterCentroid(AmberMask(Crop))
            if Found is not None:
                Cx, Cy, Size = Found
                Target = (Cx / Crop.shape[1], 1.0 - Cy / Crop.shape[0])
                print(f"{TrajId}: chain cluster {Size:.0f} px at axes "
                      f"({Target[0]:.2f}, {Target[1]:.2f})")
                Ax.annotate("chains", xy=Target, xytext=ChainText[TrajId],
                            xycoords="axes fraction",
                            textcoords="axes fraction",
                            ha="center", va="center", fontsize=8.5,
                            color="0.10", path_effects=Halo,
                            arrowprops=dict(arrowstyle="-", lw=0.7,
                                            color="0.10", shrinkA=2,
                                            shrinkB=4))
        if TrajId in Delam:
            Target, TextPos = Delam[TrajId]
            Ax.annotate("delamination", xy=Target, xytext=TextPos,
                        xycoords="axes fraction", textcoords="axes fraction",
                        ha="center", va="center", fontsize=8.5,
                        color="0.10", path_effects=Halo,
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color="0.10", shrinkA=2, shrinkB=4))

    Handles = [
        Line2D([0], [0], marker='o', ls='none', markersize=8,
               markerfacecolor=ZrCol, markeredgecolor='0.45', label='Zr'),
        Line2D([0], [0], marker='o', ls='none', markersize=5,
               markerfacecolor=CCol, markeredgecolor='none',
               label='C (lattice)'),
        Line2D([0], [0], marker='o', ls='none', markersize=6,
               markerfacecolor=ChainCol, markeredgecolor='none',
               label='C (chains)'),
        Line2D([0], [0], marker='o', ls='none', markersize=4.5,
               markerfacecolor=OCol, markeredgecolor='none', label='O'),
    ]
    Fig.legend(handles=Handles, loc='lower center', ncols=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.35, columnspacing=1.6)

    OutBase = os.path.join(ResultsDir, "Snapshots")
    Fig.savefig(OutBase + ".png", dpi=600)
    Fig.savefig(OutBase + ".pdf")
    print("saved", OutBase + ".png/.pdf")


if __name__ == "__main__":
    Main()
