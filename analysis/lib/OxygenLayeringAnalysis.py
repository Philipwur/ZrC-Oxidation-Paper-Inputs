"""
OxygenLayeringAnalysis.py
=======================================================================
Discriminates three candidate mechanisms for the oxygen layering observed in
the oxide region (periodic O density along the surface normal, diffuse Zr):

    A. generic hard-wall packing against the crystalline carbide
    B. carbide-templated order (epitaxial registry with the ZrC lattice)
    C. intrinsic smectic-like layering of the growing oxide

Three measurements decide it:
    1. layer-spacing registry          (Step 1)
    2. layering-amplitude decay        (Step 2)
    3. in-plane structure within layers (Step 3)

Reuse
-----
The streaming stack (ScanFrameOffsets / ReadFramesFrom / XyzPath and the
window + subsampling constants) is imported from ChainTopology.py - the same
final-10% window, 50 fs subsampling, seed 20260715 as all previous scripts.
The subsampled window frames are cached to
OxygenLayeringResults/FrameCache/{T}_{S}_final10_50fs.npz on first run;
re-runs skip the xyz files entirely. Existing results folders are read-only
and are not touched.

Geometry conventions
--------------------
* Surface normal = x (first lattice vector), per project convention.
* The two slab surfaces are analysed SEPARATELY ("L" = lower-x front,
  "R" = higher-x front, after re-centring the slab away from the periodic
  gas gap). Outward distance u runs from each carbide front into the oxide.
  Pooling across surfaces happens per temperature only after per-surface
  extraction (Step 3 in-plane pooling and the Step 4 verdicts).
* Interface criterion (Step 0): per frame and per side, lattice planes are
  detected as coincident Zr and C density peaks (smoothed 1D histograms,
  bin 0.10 Angstrom, Gaussian sigma 0.30 Angstrom, prominence >= 25% of the
  profile maximum, peak separation >= 1.5 Angstrom; Zr and C peaks paired
  when closer than 0.5 Angstrom). The carbide front is the outermost
  coincident plane whose spacing to its inner neighbour is within 15% of
  the median coincident-plane spacing of that frame. d_ZrC is the median
  interior coincident-plane spacing (spacings not touching a front plane,
  where >= 3 spacings exist).
* KNOWN LIMITATION carried through all reporting: the in-plane cell is
  ~9.4 x 9.4 Angstrom (2 x 2 ZrC units), so in-plane correlations are only
  measurable to r ~ 4.5 Angstrom. All in-plane conclusions are short-range
  statements; long-range order is not decidable in this cell.

Primary temperatures are 1073 and 1273 K; 873 and 973 K are run identically
but reported as thin-film controls (the verdict is not based on them).

Nothing here is stochastic; seed 20260715 is reported for consistency with
the sibling scripts.

Usage
-----
    python OxygenLayeringAnalysis.py                # full run (uses caches)
    python OxygenLayeringAnalysis.py --only 1273_3  # one trajectory
    python OxygenLayeringAnalysis.py --rebuild-cache
"""

import os
import sys
import json
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

ScriptDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")  # results and caches are kept beside the scripts in core/
sys.path.insert(0, ScriptDir)

# Reuse the streaming + window conventions from the previous scripts (import, not copy).
from ChainTopology import (
    ScanFrameOffsets, ReadFramesFrom, XyzPath,
    Temperatures, SimIndices, WindowStartPct, SubsampleFs,
)

ResultsDir = os.path.join(ScriptDir, "OxygenLayeringResults")
CacheDir = os.path.join(ResultsDir, "FrameCache")

RandomSeed = 20260715            # reported; the analysis is fully deterministic
PrimaryTemps = [1073, 1273]
ControlTemps = [873, 973]
SideLabels = ["L", "R"]

# ---- Step 0: plane detection / front bookkeeping --------------------------------
ProfileBin = 0.10                # Angstrom; <= 0.15 required by the task
PlaneSmoothSigmaAng = 0.30       # Gaussian smoothing of the Zr/C plane profiles
PlaneProminenceFrac = 0.25       # of the per-frame profile maximum
PlaneMinSeparation = 1.5         # Angstrom between detected plane peaks
CoincidenceTol = 0.50            # max |x_Zr - x_C| for a coincident (carbide) plane
FrontSpacingTolFrac = 0.15       # front plane must sit at bulk spacing within this
PlaneRefineHalfWidth = 0.60      # centroid refinement window around a plane peak

# ---- O layer detection ------------------------------------------------------------
UHistMin, UHistMax = -1.0, 15.0  # outward-distance histogram range
LayerSmoothSigmaAng = 0.15       # Gaussian smoothing of the pooled O profile
LayerProminenceFrac = 0.10       # minimum peak prominence, fraction of profile max
LayerMinSeparation = 1.2         # Angstrom between O layer peaks
LayerMinU = 0.8                  # peaks closer to the front are front-plane oxygen
LayerWindow = 0.5                # +/- window for per-frame layer membership (task)

# ---- In-plane structure -------------------------------------------------------------
G2dRLimit = 4.5                  # Angstrom (cell-size limit, see KNOWN LIMITATION)
G2dBin = 0.10
OoRdfRange = (0.8, 4.0)          # 3D O-O first-neighbour search window
OoRdfBin = 0.05
OoOxideUMin = 0.5                # O counted as "in the oxide" beyond this u
OoOxideUMax = 12.0

# ---- Verdict thresholds (Step 4; stated in Notes) -----------------------------------
SpacingMatchTolFrac = 0.06       # |d_O - reference| / reference for a "match"
RegistryMaxMeanOffset = 0.15     # mean |offset| (frac of d_ZrC) for "registered"
RegistryMaxDrift = 0.05          # |drift| (frac of d_ZrC per layer) for "registered"
DecayFastLayers = 3.0            # decay length <= this -> packing/template signature
TailLiquidRms = 0.35             # RMS(g-1) in 3.6-4.5 A below this -> decays toward 1
CrystalPeakMatchTol = 0.25       # first-peak r within this of carbide ref -> registered
CrystalMinHeight = 2.5

SpeciesCodes = {"Zr": 0, "O": 1, "C": 2, "N": 3}
SpeciesNames = {V: K for K, V in SpeciesCodes.items()}


# =============================================================================
#                     PHASE 1 - STREAM + CACHE THE WINDOW
# =============================================================================
def CachePath(Temp, Sim):
    return os.path.join(CacheDir, f"{Temp}_{Sim}_final10_50fs.npz")


def BuildTrajectoryCache(TaskArgs):
    """Stream one xyz (never loading it fully), keep the final-10% window
    subsampled every 50 fs, and store it as one small npz."""
    Temp, Sim = TaskArgs
    Path = XyzPath(Temp, Sim)
    NFrames, Offsets = ScanFrameOffsets(Path)
    IdxStart = int(NFrames * WindowStartPct)
    StartOffset = Offsets[IdxStart]

    Times, Lattices, FrameStarts = [], [], [0]
    SpeciesChunks, CoordChunks = [], []
    LastRecordedTime = None
    for TimeFs, Atoms, Coords, Lattice in ReadFramesFrom(Path, StartOffset):
        Due = LastRecordedTime is None or (TimeFs - LastRecordedTime) >= SubsampleFs
        if not Due:
            continue
        LastRecordedTime = TimeFs
        Times.append(TimeFs)
        Lattices.append(np.asarray(Lattice, dtype=np.float32))
        Codes = np.array([SpeciesCodes.get(A, 9) for A in Atoms], dtype=np.int8)
        SpeciesChunks.append(Codes)
        CoordChunks.append(Coords.astype(np.float32))
        FrameStarts.append(FrameStarts[-1] + len(Codes))

    np.savez_compressed(
        CachePath(Temp, Sim),
        Times=np.array(Times, dtype=np.float64),
        Lattices=np.stack(Lattices),
        Species=np.concatenate(SpeciesChunks),
        Coords=np.concatenate(CoordChunks),
        FrameStarts=np.array(FrameStarts, dtype=np.int64),
    )
    return Temp, Sim, len(Times)


def LoadTrajectoryCache(Temp, Sim):
    """Yield (TimeFs, SpeciesCodes, Coords float64, Lattice float64) per frame."""
    Data = np.load(CachePath(Temp, Sim))
    Starts = Data["FrameStarts"]
    for i in range(len(Data["Times"])):
        Lo, Hi = Starts[i], Starts[i + 1]
        yield (float(Data["Times"][i]), Data["Species"][Lo:Hi],
               Data["Coords"][Lo:Hi].astype(np.float64),
               Data["Lattices"][i].astype(np.float64))


# =============================================================================
#                 STEP 0 - PER-FRAME PLANES, FRONTS, COORDINATES
# =============================================================================
def SmoothedDensityPeaks(Positions, XMax, SmoothSigmaAng, ProminenceFrac, MinSepAng):
    """Peak positions of a smoothed 1D histogram of Positions on [0, XMax)."""
    NBins = int(np.ceil(XMax / ProfileBin))
    Hist, Edges = np.histogram(Positions, bins=NBins, range=(0.0, XMax))
    Smoothed = gaussian_filter1d(Hist.astype(float), SmoothSigmaAng / ProfileBin)
    if Smoothed.max() <= 0:
        return np.array([])
    PeakIdx, _ = find_peaks(
        Smoothed, prominence=ProminenceFrac * Smoothed.max(),
        distance=max(1, int(MinSepAng / ProfileBin)),
    )
    Centres = 0.5 * (Edges[:-1] + Edges[1:])
    return Centres[PeakIdx]


def AnalyseFrameGeometry(Species, Coords, Lattice):
    """
    Returns None if no usable carbide slab, else a dict with:
      XA           : re-centred x coordinate (Angstrom) per atom
      FrontL/FrontR: front plane positions
      InteriorSpacings: list of interior coincident-plane spacings
      HalfGas      : half of the outward gas extent (u cap so the two sides
                     never claim the same O atoms)
    Re-centring: the largest periodic gap in the Zr x-distribution (the gas
    region) is shifted to the cell edges so the slab is contiguous.
    """
    Inv = np.linalg.inv(Lattice)
    Frac = (Coords @ Inv) % 1.0
    Lx = float(np.linalg.norm(Lattice[0]))

    IsZr = Species == SpeciesCodes["Zr"]
    IsO = Species == SpeciesCodes["O"]
    IsC = Species == SpeciesCodes["C"]
    if IsZr.sum() < 6 or IsC.sum() < 6:
        return None

    ZrFx = np.sort(Frac[IsZr, 0])
    GapsToNext = np.diff(np.append(ZrFx, ZrFx[0] + 1.0))
    K = int(np.argmax(GapsToNext))
    GasMid = (ZrFx[K] + 0.5 * GapsToNext[K]) % 1.0
    XA = ((Frac[:, 0] - GasMid) % 1.0) * Lx

    ZrPeaks = SmoothedDensityPeaks(XA[IsZr], Lx, PlaneSmoothSigmaAng,
                                   PlaneProminenceFrac, PlaneMinSeparation)
    CPeaks = SmoothedDensityPeaks(XA[IsC], Lx, PlaneSmoothSigmaAng,
                                  PlaneProminenceFrac, PlaneMinSeparation)
    Planes = []
    for Zx in ZrPeaks:
        if len(CPeaks) == 0:
            break
        J = int(np.argmin(np.abs(CPeaks - Zx)))
        if abs(CPeaks[J] - Zx) <= CoincidenceTol:
            Rough = 0.5 * (Zx + CPeaks[J])
            Members = np.abs(XA - Rough) <= PlaneRefineHalfWidth
            Members &= (IsZr | IsC)
            Planes.append(float(XA[Members].mean()) if Members.any() else float(Rough))
    Planes = np.sort(np.array(Planes))
    if len(Planes) < 2:
        return None

    Spacings = np.diff(Planes)
    MedianSpacing = float(np.median(Spacings))

    IdxL = 0
    while (IdxL < len(Planes) - 1
           and abs(Spacings[IdxL] - MedianSpacing) > FrontSpacingTolFrac * MedianSpacing):
        IdxL += 1
    IdxR = len(Planes) - 1
    while (IdxR > 0
           and abs(Spacings[IdxR - 1] - MedianSpacing) > FrontSpacingTolFrac * MedianSpacing):
        IdxR -= 1
    if IdxR - IdxL < 1:
        return None

    KeptSpacings = np.diff(Planes[IdxL:IdxR + 1])
    Interior = KeptSpacings[1:-1] if len(KeptSpacings) >= 3 else KeptSpacings
    FrontL, FrontR = float(Planes[IdxL]), float(Planes[IdxR])
    HalfGas = 0.5 * (Lx - (FrontR - FrontL))
    InteriorPlanes = Planes[IdxL + 1:IdxR]  # strictly interior (crystal ref source)

    return {
        "XA": XA, "IsZr": IsZr, "IsO": IsO, "IsC": IsC, "Frac": Frac,
        "FrontL": FrontL, "FrontR": FrontR, "HalfGas": HalfGas,
        "InteriorSpacings": [float(S) for S in Interior],
        "InteriorPlanes": InteriorPlanes,
        "Lx": Lx, "Ly": float(Lattice[1, 1]), "Lz": float(Lattice[2, 2]),
        "Lattice": Lattice,
    }


# =============================================================================
#                     PER-TRAJECTORY EXTRACTION (Steps 0-1 data)
# =============================================================================
def ExtractTrajectory(Temp, Sim):
    """One pass over the cached frames. Returns per-surface frame data plus
    the per-frame accumulations every later step needs."""
    NBins = int(round((UHistMax - UHistMin) / ProfileBin))
    Surface = {
        Side: {"UHist": np.zeros(NBins), "Frames": [], "NFrames": 0}
        for Side in SideLabels
    }
    InteriorSpacings = []
    CrystalFrames = []       # (Y, Z, Ly, Lz) of the centre-most interior C plane
    OxideOFrames = []        # (FracSel, Lattice) for the 3D O-O first-neighbour RDF
    NoFrontFrames = 0
    NTotal = 0

    for TimeFs, Species, Coords, Lattice in LoadTrajectoryCache(Temp, Sim):
        NTotal += 1
        Geo = AnalyseFrameGeometry(Species, Coords, Lattice)
        if Geo is None:
            NoFrontFrames += 1
            continue
        InteriorSpacings.extend(Geo["InteriorSpacings"])
        XA, IsO = Geo["XA"], Geo["IsO"]
        Y, Z = Coords[:, 1], Coords[:, 2]

        OxideMask = np.zeros(len(XA), dtype=bool)
        for Side in SideLabels:
            U = (Geo["FrontL"] - XA) if Side == "L" else (XA - Geo["FrontR"])
            UCap = min(UHistMax, Geo["HalfGas"])
            Sel = IsO & (U > UHistMin) & (U < UCap)
            Rec = Surface[Side]
            Rec["UHist"] += np.histogram(
                U[Sel], bins=NBins, range=(UHistMin, UHistMax))[0]
            Rec["Frames"].append({
                "U": U[Sel].astype(np.float32),
                "Y": Y[Sel].astype(np.float32),
                "Z": Z[Sel].astype(np.float32),
            })
            Rec["NFrames"] += 1
            OxideMask |= Sel & (U > OoOxideUMin) & (U < OoOxideUMax)

        if OxideMask.sum() >= 2:
            OxideOFrames.append((Geo["Frac"][OxideMask], Geo["Lattice"]))

        if len(Geo["InteriorPlanes"]) > 0:
            Centre = 0.5 * (Geo["FrontL"] + Geo["FrontR"])
            PlaneX = Geo["InteriorPlanes"][
                int(np.argmin(np.abs(Geo["InteriorPlanes"] - Centre)))]
            CSel = Geo["IsC"] & (np.abs(XA - PlaneX) <= LayerWindow)
            if CSel.sum() >= 2:
                CrystalFrames.append((Y[CSel].copy(), Z[CSel].copy(),
                                      Geo["Ly"], Geo["Lz"]))

    return {
        "Temp": Temp, "Sim": Sim, "NTotal": NTotal, "NoFrontFrames": NoFrontFrames,
        "Surface": Surface, "InteriorSpacings": InteriorSpacings,
        "CrystalFrames": CrystalFrames, "OxideOFrames": OxideOFrames,
    }


# =============================================================================
#                   STEPS 1-2 - LAYERS, SPACING, REGISTRY, CONTRAST
# =============================================================================
def UCentres():
    NBins = int(round((UHistMax - UHistMin) / ProfileBin))
    Edges = np.linspace(UHistMin, UHistMax, NBins + 1)
    return 0.5 * (Edges[:-1] + Edges[1:])


def AnalyseSurfaceLayers(Rec, DZrC):
    """Pooled-profile O layer detection plus per-frame spacing statistics.
    Returns a dict (or None when no frame had a usable front)."""
    if Rec["NFrames"] == 0:
        return None
    Centres = UCentres()
    Profile = Rec["UHist"] / max(1, Rec["NFrames"])
    Smoothed = gaussian_filter1d(Profile, LayerSmoothSigmaAng / ProfileBin)
    if Smoothed.max() <= 0:
        return {"LayerU": [], "NFrames": Rec["NFrames"], "Profile": Smoothed}

    PeakIdx, _ = find_peaks(
        Smoothed, prominence=LayerProminenceFrac * Smoothed.max(),
        distance=max(1, int(LayerMinSeparation / ProfileBin)),
    )
    PeakIdx = [P for P in PeakIdx if Centres[P] >= LayerMinU]
    LayerU = [float(Centres[P]) for P in PeakIdx]

    Result = {"LayerU": LayerU, "NFrames": Rec["NFrames"], "Profile": Smoothed,
              "PeakIdx": PeakIdx}
    if len(LayerU) == 0:
        return Result

    # Per-frame layer positions (mean U of members within +/- LayerWindow).
    PerFrame = [[] for _ in LayerU]
    for Frame in Rec["Frames"]:
        for I, U0 in enumerate(LayerU):
            Members = np.abs(Frame["U"] - U0) <= LayerWindow
            if Members.any():
                PerFrame[I].append(float(Frame["U"][Members].mean()))
    Result["LayerPosMean"] = [float(np.mean(P)) if P else float("nan") for P in PerFrame]
    Result["LayerPosSd"] = [float(np.std(P)) if len(P) > 1 else float("nan")
                            for P in PerFrame]
    Result["LayerFrameCounts"] = [len(P) for P in PerFrame]

    # Spacings from the frame-mean layer positions; spread from frame pairs.
    Pos = np.array(Result["LayerPosMean"])
    Result["Spacings"] = [float(D) for D in np.diff(Pos)] if len(Pos) >= 2 else []
    Result["MeanSpacing"] = float(np.mean(np.diff(Pos))) if len(Pos) >= 2 else float("nan")

    # Registry vs the extrapolated carbide grid (planes at u = k * d_ZrC).
    if len(Pos) >= 1 and np.isfinite(DZrC) and DZrC > 0:
        Phase = Pos / DZrC
        Offsets = Phase - np.round(Phase)
        Result["RegistryOffsets"] = [float(O) for O in Offsets]
        Result["MeanAbsOffset"] = float(np.mean(np.abs(Offsets)))
        if len(Pos) >= 2:
            Slope = float(np.polyfit(np.arange(len(Pos)), Phase, 1)[0])
            Result["OffsetDriftPerLayer"] = Slope - 1.0
        else:
            Result["OffsetDriftPerLayer"] = float("nan")
    else:
        Result["RegistryOffsets"] = []
        Result["MeanAbsOffset"] = float("nan")
        Result["OffsetDriftPerLayer"] = float("nan")

    # Step 2 - contrast per layer from the pooled smoothed profile.
    Result["Contrast"] = ComputeLayerContrast(Smoothed, Centres, Result["PeakIdx"])
    Result["DecayLength"], Result["DecayLabel"] = FitContrastDecay(Result["Contrast"])
    return Result


def ComputeLayerContrast(Smoothed, Centres, PeakIdx):
    """C_i = (peak - trough)/(peak + trough); troughs are profile minima between
    successive peaks, averaged over the two flanks where both exist. The first
    layer's inner flank is the minimum between LayerMinU and the peak; the last
    layer uses only its inner flank (its outer flank falls into the gas and
    would bias the contrast toward 1)."""
    if len(PeakIdx) == 0:
        return []
    Troughs = []   # Troughs[i] = min between peak i and peak i+1
    for I in range(len(PeakIdx) - 1):
        Seg = Smoothed[PeakIdx[I]:PeakIdx[I + 1] + 1]
        Troughs.append(float(Seg.min()))
    InnerStart = int(np.searchsorted(Centres, LayerMinU))
    Contrast = []
    for I, P in enumerate(PeakIdx):
        Peak = float(Smoothed[P])
        Flanks = []
        if I == 0:
            if P - InnerStart >= 2:
                Flanks.append(float(Smoothed[InnerStart:P].min()))
        else:
            Flanks.append(Troughs[I - 1])
        if I < len(PeakIdx) - 1:
            Flanks.append(Troughs[I])
        if not Flanks:
            Contrast.append(float("nan"))
            continue
        Trough = float(np.mean(Flanks))
        Contrast.append((Peak - Trough) / (Peak + Trough) if Peak + Trough > 0
                        else float("nan"))
    return Contrast


def FitContrastDecay(Contrast):
    """Exponential decay length in layers from ln C_i vs (i-1).
    Returns (DecayLength, Label); needs >= 3 finite positive contrasts."""
    Vals = np.array(Contrast, dtype=float)
    Good = np.isfinite(Vals) & (Vals > 0)
    if Good.sum() < 3:
        return float("nan"), "unusable (<3 layers)"
    X = np.arange(len(Vals))[Good]
    Slope = float(np.polyfit(X, np.log(Vals[Good]), 1)[0])
    if Slope >= -1e-3:
        return float("inf"), "persistent"
    Length = -1.0 / Slope
    if Length > 2 * len(Vals):
        return Length, "persistent"
    return Length, f"{Length:.1f}"


# =============================================================================
#                    STEP 3 - IN-PLANE STRUCTURE (g2D)
# =============================================================================
def AccumulatePairHistogram2d(Y, Z, Ly, Lz, Hist, RLimit, Bin):
    """Minimum-image in-plane pair distances accumulated into Hist.
    Returns the number of pairs N(N-1)/2 (all pairs, incl. beyond RLimit)."""
    N = len(Y)
    if N < 2:
        return 0
    Dy = Y[:, None] - Y[None, :]
    Dz = Z[:, None] - Z[None, :]
    Dy -= Ly * np.round(Dy / Ly)
    Dz -= Lz * np.round(Dz / Lz)
    R = np.sqrt(Dy ** 2 + Dz ** 2)[np.triu_indices(N, k=1)]
    Hist += np.histogram(R, bins=len(Hist), range=(0.0, RLimit))[0]
    return N * (N - 1) // 2


def FinaliseG2d(Hist, NPairsSum, Area, RLimit, Bin):
    """g2D(r) = PairHist * Area / (NPairsSum * 2 pi r dr)."""
    NBins = len(Hist)
    Edges = np.linspace(0.0, RLimit, NBins + 1)
    RC = 0.5 * (Edges[:-1] + Edges[1:])
    Shell = 2.0 * np.pi * RC * Bin
    with np.errstate(divide="ignore", invalid="ignore"):
        G = Hist * Area / (max(NPairsSum, 1) * Shell)
    return RC, G


def CharacteriseG2d(RC, G, CrystalPeakR=None):
    """First-peak position/height and behaviour past the first shell.
    Rules (stated in Notes): first peak = highest local max with r >= 1.4 A;
    TailRms = RMS(g-1) over 3.6-4.5 A. 'decays toward 1' if TailRms <=
    TailLiquidRms; 'carbide-registered' if the first peak lies within
    CrystalPeakMatchTol of the carbide reference peak AND height >=
    CrystalMinHeight; otherwise 'residual structure to r limit'."""
    Sm = gaussian_filter1d(G, 1.0)
    Valid = RC >= 1.4
    if not Valid.any() or not np.isfinite(Sm[Valid]).any():
        return None
    PeakIdx, _ = find_peaks(Sm, height=0.0)
    PeakIdx = [P for P in PeakIdx if RC[P] >= 1.4]
    if not PeakIdx:
        P = int(np.nanargmax(np.where(Valid, Sm, -np.inf)))
    else:
        P = PeakIdx[int(np.argmax(Sm[PeakIdx]))]
    FirstPeakR, FirstPeakH = float(RC[P]), float(Sm[P])

    Tail = (RC >= 3.6) & (RC <= 4.5)
    TailRms = float(np.sqrt(np.nanmean((G[Tail] - 1.0) ** 2))) if Tail.any() else float("nan")
    ShortRangeO2 = bool(np.nanmax(np.where(RC < 1.4, Sm, 0.0)) > 2.0)

    if (CrystalPeakR is not None and np.isfinite(CrystalPeakR)
            and abs(FirstPeakR - CrystalPeakR) <= CrystalPeakMatchTol
            and FirstPeakH >= CrystalMinHeight):
        Behaviour = "carbide-registered short-range order"
    elif TailRms <= TailLiquidRms:
        Behaviour = "decays toward 1 (liquid-like)"
    else:
        Behaviour = "residual structure to r limit"
    return {"FirstPeakR": FirstPeakR, "FirstPeakH": FirstPeakH,
            "TailRms": TailRms, "Behaviour": Behaviour, "O2Contamination": ShortRangeO2}


# =============================================================================
#                        3D O-O PACKING REFERENCE
# =============================================================================
def OoFirstNeighbour(OxideOFrames):
    """Peak of the shell-corrected O-O pair-distance distribution among oxide
    oxygens (pooled frames). Returns (PeakR, NPairs, HasO2Peak)."""
    NBins = int(round((OoRdfRange[1] - OoRdfRange[0]) / OoRdfBin))
    Hist = np.zeros(NBins)
    NPairs = 0
    for FracSel, Lattice in OxideOFrames:
        N = len(FracSel)
        if N < 2:
            continue
        Diff = FracSel[:, None, :] - FracSel[None, :, :]
        Diff -= np.round(Diff)
        Cart = Diff @ Lattice
        R = np.linalg.norm(Cart, axis=-1)[np.triu_indices(N, k=1)]
        Hist += np.histogram(R, bins=NBins, range=OoRdfRange)[0]
        NPairs += N * (N - 1) // 2
    if Hist.sum() == 0:
        return float("nan"), 0, False
    Edges = np.linspace(OoRdfRange[0], OoRdfRange[1], NBins + 1)
    RC = 0.5 * (Edges[:-1] + Edges[1:])
    ShellCorrected = gaussian_filter1d(Hist / (4.0 * np.pi * RC ** 2 * OoRdfBin), 1.0)
    MainRegion = RC >= 1.8      # exclude any molecular O-O (~1.2-1.3 A)
    PeakR = float(RC[MainRegion][np.argmax(ShellCorrected[MainRegion])])
    O2Region = RC < 1.6
    HasO2Peak = bool(O2Region.any() and MainRegion.any()
                     and ShellCorrected[O2Region].max()
                     > 0.5 * ShellCorrected[MainRegion].max())
    return PeakR, NPairs, HasO2Peak


# =============================================================================
#                               VERDICTS (Step 4)
# =============================================================================
def SpacingRegistryVerdict(SurfaceResults, DZrC, DOo):
    """Pool usable surfaces (>= 2 layers) of one temperature."""
    Ds, Drifts, Offsets = [], [], []
    for R in SurfaceResults:
        if R and len(R.get("LayerU", [])) >= 2 and np.isfinite(R["MeanSpacing"]):
            Ds.append(R["MeanSpacing"])
            if np.isfinite(R["OffsetDriftPerLayer"]):
                Drifts.append(R["OffsetDriftPerLayer"])
            if np.isfinite(R["MeanAbsOffset"]):
                Offsets.append(R["MeanAbsOffset"])
    if not Ds:
        return "no usable surfaces", {}
    MeanD = float(np.mean(Ds))
    MeanDrift = float(np.mean(Drifts)) if Drifts else float("nan")
    MeanOffset = float(np.mean(Offsets)) if Offsets else float("nan")
    MatchZrC = abs(MeanD - DZrC) <= SpacingMatchTolFrac * DZrC
    MatchOo = np.isfinite(DOo) and abs(MeanD - DOo) <= SpacingMatchTolFrac * DOo
    Registered = (np.isfinite(MeanOffset) and MeanOffset <= RegistryMaxMeanOffset
                  and np.isfinite(MeanDrift) and abs(MeanDrift) <= RegistryMaxDrift)
    Evidence = {"MeanD": MeanD, "MeanDrift": MeanDrift, "MeanAbsOffset": MeanOffset,
                "MatchZrC": MatchZrC, "MatchOo": MatchOo, "Registered": Registered}
    if MatchZrC and Registered:
        return "B", Evidence
    if MatchZrC and not Registered:
        return "mixed (spacing ~ d_ZrC, no phase registry)", Evidence
    if MatchOo and not Registered:
        return "A", Evidence
    if not MatchZrC and not MatchOo:
        return "C", Evidence
    return "mixed", Evidence


def DecayVerdict(SurfaceResults):
    Labels = []
    for R in SurfaceResults:
        if R is None or "DecayLabel" not in R:
            continue
        if R["DecayLabel"] == "unusable (<3 layers)":
            continue
        if R["DecayLabel"] == "persistent":
            Labels.append("persistent")
        else:
            Labels.append("fast" if R["DecayLength"] <= DecayFastLayers else "slow")
    if not Labels:
        return "no usable surfaces"
    if all(L == "persistent" for L in Labels):
        return "C"
    if all(L == "fast" for L in Labels):
        return "A/B"
    return f"mixed ({Labels.count('fast')} fast / {Labels.count('slow')} slow / "\
           f"{Labels.count('persistent')} persistent)"


def InPlaneVerdict(LayerCharacterisations):
    Behaviours = [C["Behaviour"] for C in LayerCharacterisations if C]
    if not Behaviours:
        return "no usable layers"
    NReg = sum(B == "carbide-registered short-range order" for B in Behaviours)
    NLiq = sum(B == "decays toward 1 (liquid-like)" for B in Behaviours)
    if NReg == 0 and NLiq == len(Behaviours):
        return "A/C"
    if NReg == len(Behaviours):
        return "B"
    if NReg > 0:
        return f"mixed ({NReg} registered / {NLiq} liquid-like of {len(Behaviours)})"
    return f"mixed ({NLiq} liquid-like of {len(Behaviours)})"


# =============================================================================
#                              DIAGNOSTIC FIGURES
# =============================================================================
def MakeProfileFigure(AllSurfaces, OutPng):
    Fig, Axes = plt.subplots(4, 1, figsize=(8.0, 11.0), sharex=True)
    Centres = UCentres()
    for Ax, Temp in zip(Axes, Temperatures):
        for Sim in SimIndices:
            for Side in SideLabels:
                R = AllSurfaces.get((Temp, Sim, Side))
                if R is None or "Profile" not in R:
                    continue
                Line, = Ax.plot(Centres, R["Profile"], lw=0.9,
                                label=f"{Sim}{Side}")
                for U in R.get("LayerU", []):
                    Ax.axvline(U, color=Line.get_color(), lw=0.5, alpha=0.35)
        Ax.set_ylabel(f"{Temp} K\nO counts/frame/bin", fontsize=8)
        Ax.legend(fontsize=6, ncol=8, loc="upper right")
        Ax.set_xlim(UHistMin, UHistMax)
    Axes[-1].set_xlabel("u, outward distance from carbide front (Angstrom)")
    Fig.suptitle("Pooled O density profiles per surface (vertical lines = detected layers)",
                 fontsize=10)
    Fig.tight_layout()
    Fig.savefig(OutPng, dpi=200)
    plt.close(Fig)


def MakeG2dFigure(G2dByTempLayer, CrystalCurves, OutPng):
    Fig, Axes = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=True, sharey=True)
    for Ax, Temp in zip(Axes.ravel(), Temperatures):
        for LayerIdx, (RC, G, NFrames) in sorted(G2dByTempLayer.get(Temp, {}).items()):
            Ax.plot(RC, G, lw=1.2, label=f"layer {LayerIdx} ({NFrames} fr)")
        if Temp in CrystalCurves:
            RC, G, _ = CrystalCurves[Temp]
            Ax.plot(RC, G, "k--", lw=1.0, label="carbide C plane")
        Ax.axhline(1.0, color="0.7", lw=0.7)
        Ax.set_title(f"{Temp} K", fontsize=10)
        Ax.set_xlim(0, G2dRLimit)
        Ax.set_ylim(0, 6)
        Ax.legend(fontsize=7)
    for Ax in Axes[-1]:
        Ax.set_xlabel("r (Angstrom)")
    for Ax in Axes[:, 0]:
        Ax.set_ylabel("g2D(r)")
    Fig.suptitle("In-plane pair correlation per O layer (r limit 4.5 A = cell size)",
                 fontsize=10)
    Fig.tight_layout()
    Fig.savefig(OutPng, dpi=200)
    plt.close(Fig)


# =============================================================================
#                                   DRIVER
# =============================================================================
def Main():
    Parser = argparse.ArgumentParser(description="Oxygen layering mechanism discrimination.")
    Parser.add_argument("--jobs", type=int,
                        default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    Parser.add_argument("--only", type=str, default=None)
    Parser.add_argument("--rebuild-cache", action="store_true")
    Args = Parser.parse_args()

    os.makedirs(ResultsDir, exist_ok=True)
    os.makedirs(CacheDir, exist_ok=True)

    Tasks = ([tuple(int(x) for x in Args.only.split("_"))] if Args.only
             else [(T, S) for T in Temperatures for S in SimIndices])

    # ---- Phase 1: build missing caches (parallel; the only xyz access) ----------
    Missing = [T for T in Tasks
               if Args.rebuild_cache or not os.path.exists(CachePath(*T))]
    ReStreamed = len(Missing) > 0
    if Missing:
        print(f"Caching {len(Missing)} trajectory windows (jobs={Args.jobs}) ...",
              flush=True)
        if Args.jobs > 1 and len(Missing) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=Args.jobs) as Pool:
                Futures = {Pool.submit(BuildTrajectoryCache, T): T for T in Missing}
                for Fut in as_completed(Futures):
                    Temp, Sim, N = Fut.result()
                    print(f"  cached {Temp}_{Sim}: {N} frames", flush=True)
        else:
            for T in Missing:
                Temp, Sim, N = BuildTrajectoryCache(T)
                print(f"  cached {Temp}_{Sim}: {N} frames", flush=True)
    else:
        print("All trajectory window caches present; no xyz access needed.", flush=True)

    # ---- Phase 2: extraction ------------------------------------------------------
    Extractions = {}
    for Temp, Sim in Tasks:
        Extractions[(Temp, Sim)] = ExtractTrajectory(Temp, Sim)
        E = Extractions[(Temp, Sim)]
        print(f"  {Temp}_{Sim}: {E['NTotal']} frames "
              f"({E['NoFrontFrames']} without usable front)", flush=True)

    # ---- d_ZrC internal reference --------------------------------------------------
    DZrCByTemp = {}
    for Temp in Temperatures:
        Spacings = [S for (T, _), E in Extractions.items() if T == Temp
                    for S in E["InteriorSpacings"]]
        DZrCByTemp[Temp] = float(np.median(Spacings)) if Spacings else float("nan")
    AllSpacings = [S for E in Extractions.values() for S in E["InteriorSpacings"]]
    DZrCGlobal = float(np.median(AllSpacings)) if AllSpacings else float("nan")

    # ---- O-O packing reference (3D, oxide region) -----------------------------------
    OoByTemp = {}
    for Temp in Temperatures:
        Frames = [F for (T, _), E in Extractions.items() if T == Temp
                  for F in E["OxideOFrames"]]
        OoByTemp[Temp] = OoFirstNeighbour(Frames)
    PrimaryOoFrames = [F for (T, _), E in Extractions.items() if T in PrimaryTemps
                       for F in E["OxideOFrames"]]
    DOoPrimary, DOoPairs, DOoHasO2 = OoFirstNeighbour(PrimaryOoFrames)

    # ---- Steps 1-2 per surface -------------------------------------------------------
    AllSurfaces = {}
    for (Temp, Sim), E in Extractions.items():
        for Side in SideLabels:
            AllSurfaces[(Temp, Sim, Side)] = AnalyseSurfaceLayers(
                E["Surface"][Side], DZrCByTemp[Temp])

    # ---- Step 3: in-plane structure, pooled per (temperature, layer index) -----------
    Area = None
    G2dAccum = {}       # (Temp, LayerIdx) -> [Hist, NPairs, FrameCount]
    NBinsG = int(round(G2dRLimit / G2dBin))
    for (Temp, Sim), E in Extractions.items():
        # In-plane cell from the trajectory's first cached lattice.
        Ly, Lz = LyLzOf(E)
        if Area is None:
            Area = Ly * Lz
        for Side in SideLabels:
            R = AllSurfaces[(Temp, Sim, Side)]
            if R is None or not R.get("LayerU"):
                continue
            for Frame in E["Surface"][Side]["Frames"]:
                for LayerIdx, U0 in enumerate(R["LayerU"], start=1):
                    Members = np.abs(Frame["U"] - U0) <= LayerWindow
                    if Members.sum() < 2:
                        continue
                    Key = (Temp, LayerIdx)
                    if Key not in G2dAccum:
                        G2dAccum[Key] = [np.zeros(NBinsG), 0, 0]
                    NP = AccumulatePairHistogram2d(
                        Frame["Y"][Members].astype(np.float64),
                        Frame["Z"][Members].astype(np.float64),
                        Ly, Lz, G2dAccum[Key][0], G2dRLimit, G2dBin)
                    G2dAccum[Key][1] += NP
                    G2dAccum[Key][2] += 1

    CrystalAccum = {}   # Temp -> [Hist, NPairs, FrameCount]
    for (Temp, Sim), E in Extractions.items():
        for (Y, Z, FLy, FLz) in E["CrystalFrames"]:
            if Temp not in CrystalAccum:
                CrystalAccum[Temp] = [np.zeros(NBinsG), 0, 0]
            NP = AccumulatePairHistogram2d(
                Y.astype(np.float64), Z.astype(np.float64), FLy, FLz,
                CrystalAccum[Temp][0], G2dRLimit, G2dBin)
            CrystalAccum[Temp][1] += NP
            CrystalAccum[Temp][2] += 1

    if Area is None:
        Area = 9.456 * 9.462     # fallback, never expected

    CrystalCurves, CrystalChar = {}, {}
    for Temp, (Hist, NPairs, NFrames) in CrystalAccum.items():
        RC, G = FinaliseG2d(Hist, NPairs, Area, G2dRLimit, G2dBin)
        CrystalCurves[Temp] = (RC, G, NFrames)
        CrystalChar[Temp] = CharacteriseG2d(RC, G)

    PrimaryCrystalPeaks = [CrystalChar[T]["FirstPeakR"] for T in PrimaryTemps
                           if T in CrystalChar and CrystalChar[T]]
    CrystalPeakR = float(np.mean(PrimaryCrystalPeaks)) if PrimaryCrystalPeaks else float("nan")

    G2dByTempLayer, G2dChar = {}, {}
    for (Temp, LayerIdx), (Hist, NPairs, NFrames) in G2dAccum.items():
        RC, G = FinaliseG2d(Hist, NPairs, Area, G2dRLimit, G2dBin)
        G2dByTempLayer.setdefault(Temp, {})[LayerIdx] = (RC, G, NFrames)
        G2dChar[(Temp, LayerIdx)] = CharacteriseG2d(RC, G, CrystalPeakR)

    # ---- Step 4 verdicts ----------------------------------------------------------------
    Verdicts = {}
    for Temp in Temperatures:
        SurfList = [AllSurfaces[(Temp, S, Side)] for S in SimIndices
                    for Side in SideLabels if (Temp, S, Side) in AllSurfaces]
        SpacingV, SpacingEvidence = SpacingRegistryVerdict(
            SurfList, DZrCByTemp[Temp], OoByTemp[Temp][0])
        Verdicts[Temp] = {
            "Spacing": SpacingV, "SpacingEvidence": SpacingEvidence,
            "Decay": DecayVerdict(SurfList),
            "InPlane": InPlaneVerdict(
                [G2dChar.get((Temp, L)) for L in sorted(
                    LI for (T, LI) in G2dChar if T == Temp)]),
        }

    # ---- Figures, JSON, SUMMARY ---------------------------------------------------------
    MakeProfileFigure(AllSurfaces, os.path.join(ResultsDir, "OxygenProfiles.png"))
    MakeG2dFigure(G2dByTempLayer, CrystalCurves,
                  os.path.join(ResultsDir, "InPlaneG2d.png"))

    Payload = BuildJsonPayload(
        Extractions, AllSurfaces, DZrCByTemp, DZrCGlobal, OoByTemp,
        DOoPrimary, DOoPairs, DOoHasO2, G2dChar, G2dByTempLayer,
        CrystalChar, CrystalCurves, Verdicts, ReStreamed)
    with open(os.path.join(ResultsDir, "OxygenLayeringResults.json"), "w") as F:
        json.dump(Payload, F, indent=2, default=JsonSafe)

    WriteSummary(Payload, AllSurfaces, Extractions, G2dChar, CrystalChar, Verdicts)
    print("\nDone. Results in", ResultsDir, flush=True)


def LyLzOf(Extraction):
    """In-plane cell lengths (diagonal approximation) from any cached frame."""
    Temp, Sim = Extraction["Temp"], Extraction["Sim"]
    Data = np.load(CachePath(Temp, Sim))
    L = Data["Lattices"][0]
    return float(L[1, 1]), float(L[2, 2])


def JsonSafe(X):
    if isinstance(X, (np.floating, np.integer)):
        return float(X)
    if isinstance(X, np.ndarray):
        return X.tolist()
    return str(X)


# =============================================================================
#                              REPORTING
# =============================================================================
def SurfaceName(Sim, Side):
    return f"{Sim}{Side}"


def BuildJsonPayload(Extractions, AllSurfaces, DZrCByTemp, DZrCGlobal, OoByTemp,
                     DOoPrimary, DOoPairs, DOoHasO2, G2dChar, G2dByTempLayer,
                     CrystalChar, CrystalCurves, Verdicts, ReStreamed):
    Surfaces = {}
    for (Temp, Sim, Side), R in AllSurfaces.items():
        Key = f"{Temp}_{Sim}_{Side}"
        if R is None:
            Surfaces[Key] = None
            continue
        Surfaces[Key] = {K: V for K, V in R.items()
                         if K not in ("Profile", "PeakIdx")}
    return {
        "Seed": RandomSeed,
        "Subsampling_fs": SubsampleFs,
        "Window": "final 10% of frames (WindowStartPct 0.90, from ChainTopology)",
        "Source": "FrameCache reused" if not ReStreamed else "xyz re-streamed then cached",
        "InterfaceCriterion": (
            "outermost coincident Zr/C density-peak plane (peaks paired within "
            f"{CoincidenceTol} A) whose spacing to its inner neighbour is within "
            f"{int(FrontSpacingTolFrac*100)}% of the frame's median plane spacing"),
        "DZrC_ByTemp": DZrCByTemp,
        "DZrC_Global": DZrCGlobal,
        "OoFirstNeighbour_ByTemp": {
            str(T): {"PeakR": V[0], "NPairs": V[1], "MolecularO2Detected": V[2]}
            for T, V in OoByTemp.items()},
        "OoFirstNeighbour_Primary": {
            "PeakR": DOoPrimary, "NPairs": DOoPairs, "MolecularO2Detected": DOoHasO2},
        "FrameCounts": {f"{T}_{S}": {"Total": E["NTotal"],
                                     "NoUsableFront": E["NoFrontFrames"]}
                        for (T, S), E in Extractions.items()},
        "Surfaces": Surfaces,
        "InPlane": {f"{T}_layer{L}": C for (T, L), C in G2dChar.items()},
        "InPlaneFrameCounts": {
            f"{T}_layer{L}": G2dByTempLayer[T][L][2]
            for T in G2dByTempLayer for L in G2dByTempLayer[T]},
        "CrystalReference": {str(T): C for T, C in CrystalChar.items()},
        "CrystalReferenceFrames": {str(T): CrystalCurves[T][2] for T in CrystalCurves},
        "Verdicts": {str(T): {K: V for K, V in D.items() if K != "SpacingEvidence"}
                     for T, D in Verdicts.items()},
        "VerdictEvidence": {str(T): D["SpacingEvidence"] for T, D in Verdicts.items()},
        "Thresholds": {
            "SpacingMatchTolFrac": SpacingMatchTolFrac,
            "RegistryMaxMeanOffset": RegistryMaxMeanOffset,
            "RegistryMaxDrift": RegistryMaxDrift,
            "DecayFastLayers": DecayFastLayers,
            "TailLiquidRms": TailLiquidRms,
            "CrystalPeakMatchTol": CrystalPeakMatchTol,
            "CrystalMinHeight": CrystalMinHeight,
        },
    }


def Fmt(X, Nd=2):
    if X is None or (isinstance(X, float) and not np.isfinite(X)):
        return "-"
    return f"{X:.{Nd}f}"


def WriteSummary(Payload, AllSurfaces, Extractions, G2dChar, CrystalChar, Verdicts):
    DZrCGlobal = Payload["DZrC_Global"]
    DOo = Payload["OoFirstNeighbour_Primary"]["PeakR"]

    # Usable-surface bookkeeping (usable = valid front frames and >= 2 O layers).
    UsableLine = []
    for Temp in Temperatures:
        Usable, Excluded = 0, []
        for Sim in SimIndices:
            for Side in SideLabels:
                R = AllSurfaces.get((Temp, Sim, Side))
                if R is None:
                    Excluded.append(f"{SurfaceName(Sim, Side)}: no usable front")
                elif len(R.get("LayerU", [])) < 2:
                    Excluded.append(
                        f"{SurfaceName(Sim, Side)}: {len(R.get('LayerU', []))} O layer(s)")
                else:
                    Usable += 1
        Part = f"{Temp} K: {Usable}/8"
        if Excluded:
            Part += " (" + "; ".join(Excluded) + ")"
        UsableLine.append(Part)

    L = []
    L.append("# Oxygen layering: registry, decay, and in-plane structure")
    L.append("")
    L.append(f"Source: {Payload['Source']}, seed {RandomSeed}, subsampling "
             f"{int(SubsampleFs)} fs")
    L.append(f"Interface criterion: {Payload['InterfaceCriterion']}  |  "
             f"d_ZrC (internal reference): {Fmt(DZrCGlobal)} Angstrom")
    L.append("Surfaces analysed per temperature: " + " | ".join(UsableLine))
    L.append("")
    L.append("## 1. Layer spacings and registry")
    L.append("")
    L.append("| T (K) | Surface | Layers resolved | Mean d_O (Ang) | d_O / d_ZrC | "
             "Offset drift (frac of d_ZrC per layer) |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for Temp in Temperatures:
        DZrCT = Payload["DZrC_ByTemp"][Temp]
        for Sim in SimIndices:
            for Side in SideLabels:
                R = AllSurfaces.get((Temp, Sim, Side))
                if R is None:
                    L.append(f"| {Temp} | {SurfaceName(Sim, Side)} | 0 | - | - | - |")
                    continue
                NL = len(R.get("LayerU", []))
                Ratio = (R["MeanSpacing"] / DZrCT
                         if NL >= 2 and np.isfinite(R["MeanSpacing"])
                         and np.isfinite(DZrCT) else float("nan"))
                Drift = R.get("OffsetDriftPerLayer", float("nan"))
                L.append(f"| {Temp} | {SurfaceName(Sim, Side)} | {NL} | "
                         f"{Fmt(R.get('MeanSpacing', float('nan')))} | {Fmt(Ratio)} | "
                         f"{Fmt(Drift, 3) if NL >= 2 else '-'} |")
    L.append("")
    L.append(f"O-O first-neighbour distance in oxide (packing reference): "
             f"{Fmt(DOo)} Angstrom")
    L.append("")
    L.append("## 2. Amplitude decay")
    L.append("")
    L.append("| T (K) | Surface | Contrast by layer (C1, C2, C3, ...) | "
             "Decay length (layers) or \"persistent\" |")
    L.append("|---:|---|---|---|")
    for Temp in Temperatures:
        for Sim in SimIndices:
            for Side in SideLabels:
                R = AllSurfaces.get((Temp, Sim, Side))
                if R is None or not R.get("Contrast"):
                    L.append(f"| {Temp} | {SurfaceName(Sim, Side)} | - | "
                             "unusable (<3 layers) |")
                    continue
                CStr = ", ".join(Fmt(C) for C in R["Contrast"])
                L.append(f"| {Temp} | {SurfaceName(Sim, Side)} | {CStr} | "
                         f"{R['DecayLabel']} |")
    L.append("")
    L.append("## 3. In-plane structure (r limited to 4.5 Ang by cell size)")
    L.append("")
    L.append("| T (K) | Layer index | First-peak r (Ang) | Peak height | "
             "Behaviour past first shell | Frames |")
    L.append("|---:|---:|---:|---:|---|---:|")
    for Temp in Temperatures:
        LayerIdxs = sorted(LI for (T, LI) in G2dChar if T == Temp)
        for LI in LayerIdxs:
            C = G2dChar[(Temp, LI)]
            NFrames = Payload["InPlaneFrameCounts"].get(f"{Temp}_layer{LI}", 0)
            if C is None:
                L.append(f"| {Temp} | {LI} | - | - | too few pairs | {NFrames} |")
                continue
            L.append(f"| {Temp} | {LI} | {Fmt(C['FirstPeakR'])} | "
                     f"{Fmt(C['FirstPeakH'], 1)} | {C['Behaviour']} | {NFrames} |")
    L.append("")
    PrimaryRef = [CrystalChar[T] for T in PrimaryTemps
                  if T in CrystalChar and CrystalChar[T]]
    if PrimaryRef:
        RefR = float(np.mean([C["FirstPeakR"] for C in PrimaryRef]))
        RefH = float(np.mean([C["FirstPeakH"] for C in PrimaryRef]))
        RefOsc = ("yes" if any(C["TailRms"] > TailLiquidRms for C in PrimaryRef)
                  else "no")
    else:
        RefR, RefH, RefOsc = float("nan"), float("nan"), "-"
    L.append(f"Crystalline reference (carbide C plane): first-peak r {Fmt(RefR)}, "
             f"height {Fmt(RefH, 1)}, oscillations to r limit: {RefOsc}")
    L.append("")
    L.append("## 4. Verdict per interpretation rule")
    L.append("")
    L.append("| Measurement | 1073 K | 1273 K | (873/973 K controls) |")
    L.append("|---|---|---|---|")
    ControlOf = lambda Key: (f"{Verdicts[873][Key]} / {Verdicts[973][Key]}")
    L.append(f"| Spacing/registry | {Verdicts[1073]['Spacing']} | "
             f"{Verdicts[1273]['Spacing']} | {ControlOf('Spacing')} |")
    L.append(f"| Amplitude decay | {Verdicts[1073]['Decay']} | "
             f"{Verdicts[1273]['Decay']} | {ControlOf('Decay')} |")
    L.append(f"| In-plane structure | {Verdicts[1073]['InPlane']} | "
             f"{Verdicts[1273]['InPlane']} | {ControlOf('InPlane')} |")
    L.append("")
    L.append("## 5. Notes")
    for N in BuildNotes(Payload, AllSurfaces, Extractions):
        L.append(f"- {N}")

    Text = "\n".join(L)
    with open(os.path.join(ResultsDir, "SUMMARY.md"), "w") as F:
        F.write(Text + "\n")
    print(Text)


def BuildNotes(Payload, AllSurfaces, Extractions):
    Notes = []
    Notes.append(
        f"Peak detection - carbide planes: per-frame Zr and C histograms "
        f"(bin {ProfileBin} A) smoothed with a {PlaneSmoothSigmaAng} A Gaussian; "
        f"scipy find_peaks with prominence >= {int(PlaneProminenceFrac*100)}% of the "
        f"profile max and >= {PlaneMinSeparation} A separation; Zr/C peaks paired "
        f"within {CoincidenceTol} A; plane position refined as the Zr+C centroid "
        f"within +/- {PlaneRefineHalfWidth} A. O layers: pooled per-surface profile "
        f"(front-aligned per frame, bin {ProfileBin} A <= 0.15 A), smoothed "
        f"{LayerSmoothSigmaAng} A, find_peaks with minimum prominence "
        f"{int(LayerProminenceFrac*100)}% of profile max and >= "
        f"{LayerMinSeparation} A separation; peaks with u < {LayerMinU} A are "
        f"treated as front-plane oxygen, not layers.")
    Notes.append(
        "Layer positions/spacings: per-frame layer position = mean u of O atoms "
        f"within +/- {LayerWindow} A of the pooled peak; frame-to-frame SD of these "
        "positions is the quoted uncertainty source (per-surface values in "
        "OxygenLayeringResults.json). Contrast uses the pooled smoothed profile; "
        "the last layer's outer flank falls into the gas, so its contrast uses the "
        "inner flank only (bias toward 1 avoided).")
    Notes.append(
        "Registry: offsets are (u_i / d_ZrC) minus the nearest integer with the "
        "front plane as phase zero; drift is the slope of u_i/d_ZrC vs layer index "
        "minus 1 (a template predicts drift ~ 0 AND small offsets; both thresholds "
        f"in JSON: mean |offset| <= {RegistryMaxMeanOffset}, |drift| <= "
        f"{RegistryMaxDrift}).")
    Notes.append(
        f"Verdict thresholds: spacing 'match' = within {int(SpacingMatchTolFrac*100)}% "
        f"of the reference; decay 'fast' = decay length <= {DecayFastLayers:.0f} "
        f"layers; in-plane 'liquid-like' = RMS(g2D-1) over 3.6-4.5 A <= "
        f"{TailLiquidRms}; 'carbide-registered' = first peak within "
        f"{CrystalPeakMatchTol} A of the carbide reference peak and height >= "
        f"{CrystalMinHeight}.")
    Notes.append(
        "KNOWN LIMITATION: the in-plane cell is ~9.4 x 9.4 A (2 x 2 ZrC units), so "
        "g2D(r) is only measurable to r ~ 4.5 A and every in-plane statement here "
        "is short-range only; long-range order is not decidable in this cell. The "
        "carbide reference itself shows only its first shell (~3.3 A) inside the "
        "r limit - 'oscillations to r limit' distinguishes a sharp, deep-troughed "
        "single shell from a liquid-like tail, not true long-range order.")
    Notes.append(
        "In-plane minimum image uses the diagonal in-plane cell (off-diagonal "
        "lattice terms ~0.01 A are neglected).")
    NoFront = {f"{T}_{S}": E["NoFrontFrames"] for (T, S), E in Extractions.items()
               if E["NoFrontFrames"] > 0}
    Notes.append(
        f"Frames without a usable carbide front: {NoFront if NoFront else 'none'}. "
        "Surfaces with < 2 resolved O layers are excluded from Step 1 pooling; "
        "< 3 layers are flagged unusable for Step 2 (they still contribute their "
        "resolved layers to Step 3).")
    O2Layers = [K for K, C in Payload["InPlane"].items()
                if C and C.get("O2Contamination")]
    Notes.append(
        "Molecular O2 contamination (g2D peak below 1.4 A) flagged in layers: "
        f"{O2Layers if O2Layers else 'none'}; molecular O-O (< 1.6 A) is excluded "
        "from the 3D packing reference.")
    Notes.append(
        "Two structures the peak detector cannot separate from oxide layers, "
        "both left in the tables per the operational definition (maxima outward "
        "of the front) and called out here instead: (i) several surfaces show a "
        "weak front-adjacent O peak at u ~ 1.0-1.2 A (oxygen bonded into the "
        "outermost carbide plane region); where detected it becomes layer 1 and "
        "pulls that surface's mean d_O down (the low-d_O outliers in Section 1, "
        "e.g. 973 3L, 1273 4L) - interior spacings cluster at ~2.4 A regardless; "
        "(ii) the outermost 'layer' of the O2-flagged rows in Section 3 (first "
        "peak ~1.45 A) is a molecular-O2 gas adlayer, not oxide - it inflates "
        "the layer count and the last contrast value of the affected surfaces.")
    Notes.append(
        "Sampling limits: 50 fs subsampling gives ~260 frames per trajectory in "
        "the final-10% window (~13 ps); layer positions drift with the advancing "
        "front during the window, which smears pooled profiles by roughly the "
        "front's per-window advance - the per-frame front alignment removes the "
        "common-mode part but not differential relaxation between layers. "
        "Frame counts per in-plane row are in Section 3; rows with few frames "
        "(controls, deep layers) are correspondingly soft.")
    Notes.append(
        "873/973 K are thin-film controls (oxide often < 3 layers); the verdict "
        "column for controls is reported but, per protocol, carries no weight in "
        "the mechanism call.")
    Notes.append(
        "Nothing stochastic is computed; seed 20260715 is quoted for consistency "
        "with the sibling scripts. Existing results folders were read-only inputs; "
        "all outputs live in OxygenLayeringResults/.")
    return Notes


if __name__ == "__main__":
    Main()
