"""
SnapshotsRender.py
=======================================================================
Stage 1 of the Snapshots figure: ray-traced OVITO renders of the 140 ps configurations
of the two cells that carry the paper's crossover contrast.

Selection (recorded in the caption): (a) 973_3, the cell whose chain
commitment count lies nearest the four-cell median (EventTableV2
counts 10/6/8/13, median 7.5); (b) 1273_3, the cell with the LARGEST
chain reservoir (counts 12/15/26/16), chosen so
the displacement of the oxide is actually visible. 1273_3 is the
artifact-censored trajectory (three carbons deleted by run-era
bookkeeping); the deletion does not affect the rendered configuration.

Input frames: SnapshotsResults/Frame140_{traj}.xyz, single extended-xyz
frames sliced read-only from the Refresh140 source trajectories at the
sample nearest 140.000 ps (973_3: 140000.0 fs, 1273_4: 140000.4 fs).

Chain carbon = instantaneous chain state at the rendered frame,
>= 1 carbon neighbour within CutoffCC = 2.31 A (BootstrapCalculation
convention), minimum image. Counts are printed for the sanity check
against the paper's confirmed chain counts (973_3: 8, 1273_4: 15).

Run in an environment that has the ovito Python package:
  python SnapshotsRender.py
Output: SnapshotsResults/Panel_{traj}.png (white background, no text;
stage 2 = SnapshotsFigure.py adds typography at print scale).
"""

import os
import warnings

warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

import numpy as np

from ovito.io import import_file
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CreateBondsModifier
from ovito.vis import Viewport, TachyonRenderer
from ovito.data import ParticleType

ScriptDir = os.path.dirname(os.path.abspath(__file__))
ResultsDir = os.path.join(ScriptDir, "SnapshotsResults")

Trajectories = ["973_3", "1273_3"]
CutoffCC = 2.31          # A, BootstrapCalculation convention
PanelW, PanelH = 4800, 1140

Style = {'Zr': ((0.72, 0.75, 0.80), 1.30),
         'C': ((0.15, 0.15, 0.15), 0.55),
         'O': ((0.82, 0.12, 0.10), 0.45)}    # vibrant red O
ChainStyle = ((0.90, 0.60, 0.05), 0.75)      # amber, enlarged


def MinImageCross(FracA, FracB, Cell):
    """Min-image distances |A_i - B_j| for fractional coords, general cell."""
    Delta = FracA[:, None, :] - FracB[None, :, :]
    Delta -= np.round(Delta)
    Cart = Delta @ Cell
    return np.linalg.norm(Cart, axis=2)


def RenderOne(TrajId):
    Path = os.path.join(ResultsDir, f"Frame140_{TrajId}.xyz")
    Src = import_file(Path)
    Data = Src.compute(0)

    Cell = np.array(Data.cell[:, :3]).T          # rows = cell vectors
    Pos = np.array(Data.particles['Position'])
    Frac = Pos @ np.linalg.inv(Cell)
    Frac = np.mod(Frac, 1.0)

    TypeProp = Data.particles_['Particle Type_']
    Names = {T.id: T.name for T in TypeProp.types}
    Species = np.array([Names[I] for I in np.array(TypeProp)])

    # Slab-contiguous shift along the surface normal (cell vector 0),
    # house method: centre the largest Zr gap at the periodic boundary.
    ZrX = np.sort(Frac[Species == 'Zr', 0])
    Gaps = np.diff(np.concatenate([ZrX, [ZrX[0] + 1.0]]))
    Shift = ZrX[np.argmax(Gaps)] + Gaps.max() / 2.0
    Frac[:, 0] = np.mod(Frac[:, 0] - Shift, 1.0)
    Data.particles_['Position_'][:] = Frac @ Cell

    # Chain carbon: >= 1 C neighbour within CutoffCC, minimum image.
    CRows = np.where(Species == 'C')[0]
    DCC = MinImageCross(Frac[CRows], Frac[CRows], Cell)
    np.fill_diagonal(DCC, np.inf)
    ChainMask = (DCC < CutoffCC).any(axis=1)
    ChainRows = CRows[ChainMask]
    print(f"{TrajId}: {len(ChainRows)} chain carbons (instantaneous, "
          f"{len(CRows)} C in cell)")

    # Recentre in-plane (cell vector 2, the render's vertical) on the
    # circular mean of the chain carbons, so strands are not cut by the
    # periodic boundary in the printed panel.
    if len(ChainRows):
        Ang = 2.0 * np.pi * Frac[ChainRows, 2]
        Mean = np.arctan2(np.sin(Ang).mean(), np.cos(Ang).mean()) / (2 * np.pi)
        Frac[:, 2] = np.mod(Frac[:, 2] - Mean + 0.5, 1.0)
        Data.particles_['Position_'][:] = Frac @ Cell

    for T in list(TypeProp.types):
        if T.name in Style:
            Col, Rad = Style[T.name]
            TM = TypeProp.make_mutable(T)
            TM.color = Col
            TM.radius = Rad
    NextId = max(T.id for T in TypeProp.types) + 1
    TChain = ParticleType(id=NextId, name='Cchain',
                          color=ChainStyle[0], radius=ChainStyle[1])
    TypeProp.types.append(TChain)
    Arr = np.array(TypeProp)
    Arr[ChainRows] = TChain.id
    Data.particles_['Particle Type_'][:] = Arr

    Pipe = Pipeline(source=StaticSource(data=Data))
    Bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    Bonds.set_pairwise_cutoff('Cchain', 'Cchain', CutoffCC)
    Bonds.vis.width = 0.30
    Pipe.modifiers.append(Bonds)
    Data.cell.vis.enabled = False
    Pipe.add_to_scene()

    Vp = Viewport(type=Viewport.Type.Ortho, camera_dir=(0, -1, 0))
    Vp.zoom_all(size=(PanelW, PanelH))
    Out = os.path.join(ResultsDir, f"Panel_{TrajId}.png")
    Vp.render_image(filename=Out, size=(PanelW, PanelH),
                    renderer=TachyonRenderer(shadows=False,
                                             direct_light_intensity=1.05),
                    background=(1, 1, 1))
    Pipe.remove_from_scene()
    print("rendered", Out)


def Main():
    for TrajId in Trajectories:
        RenderOne(TrajId)


if __name__ == "__main__":
    Main()
