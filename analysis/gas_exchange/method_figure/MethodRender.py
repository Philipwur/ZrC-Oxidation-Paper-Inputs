"""OVITO render of the cell for the method figure.

Stage 1 of two. Same renderer, camera, colours and radii as the Snapshots
figure of the Discussion (core/SnapshotsRender.py), without the
chain colouring. Reads MethodFrame.xyz, written by MethodFigure.py, and writes
Panel_Method.png (white background, no text). The two small magenta markers
give MethodFigure.py the scale of the picture and are painted out there.

Run in an environment that has the ovito Python package:
    python MethodRender.py
"""

import os
import warnings

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import numpy as np
from ovito.io import import_file
from ovito.pipeline import Pipeline, StaticSource
from ovito.vis import TachyonRenderer, Viewport

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_WIDTH = 4800

STYLE = {"Zr": ((0.72, 0.75, 0.80), 1.30),
         "C": ((0.15, 0.15, 0.15), 0.55),
         "O": ((0.82, 0.12, 0.10), 0.45),
         "Mk": ((1.0, 0.0, 1.0), 0.30)}


def main():
    source = import_file(os.path.join(HERE, "MethodFrame.xyz"))
    data = source.compute(0)
    types = data.particles_["Particle Type_"]
    for particle_type in list(types.types):
        if particle_type.name in STYLE:
            colour, radius = STYLE[particle_type.name]
            mutable = types.make_mutable(particle_type)
            mutable.color = colour
            mutable.radius = radius

    position = np.array(data.particles["Position"])
    extent = position.max(axis=0) - position.min(axis=0)
    height = int(round(PANEL_WIDTH * (extent[2] + 3.0) / (extent[0] + 3.0)))

    pipeline = Pipeline(source=StaticSource(data=data))
    data.cell.vis.enabled = False
    pipeline.add_to_scene()
    viewport = Viewport(type=Viewport.Type.Ortho, camera_dir=(0, -1, 0))
    viewport.zoom_all(size=(PANEL_WIDTH, height))
    out = os.path.join(HERE, "Panel_Method.png")
    viewport.render_image(filename=out, size=(PANEL_WIDTH, height),
                          renderer=TachyonRenderer(shadows=False, direct_light_intensity=1.05),
                          background=(1, 1, 1))
    pipeline.remove_from_scene()
    print("rendered", out, (PANEL_WIDTH, height))


if __name__ == "__main__":
    main()
