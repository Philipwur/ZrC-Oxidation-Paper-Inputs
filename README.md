# Input files and analysis scripts for "Illuminating early ZrC oxidation using ensemble small-cell DFT with dynamic gas management"

This repository holds the input files of the simulations in the paper and the scripts that produce its figures and tables. The simulations were run with the SGUSCHI framework (https://github.com/Philipwur/SGUSCHI) and VASP 6.5.1.

```
inputs/      input files of the simulations
analysis/    analysis scripts
```

## Input files

| File | What it is |
|---|---|
| `inputs/POSCAR` | Starting structure of one trajectory (873 K, cell 3). A ZrC(100) slab of 48 Zr and 48 C atoms in six planes, with 10 O2 molecules in the gas volume |
| `inputs/INCAR` | VASP settings. A template, see below |
| `inputs/job.in` | Settings of the small-cell workflow |
| `inputs/OxParams` | Settings of the gas exchange |
| `inputs/CovalentRadii` | Covalent radii used to detect gas molecules |

### How the 16 trajectories were set up

- Four temperatures (873, 973, 1073 and 1273 K) with four cells each.
- All 16 cells start from the same slab. The 10 O2 molecules are placed by the framework at the positions furthest from all atoms, with a random search, so their positions differ between the cells. The `POSCAR` here is the start of one of them.
- Initial velocities are drawn by VASP for the target temperature.

### Settings that the workflow sets itself

The `INCAR` is the template given to SGUSCHI. SGUSCHI sets the tags below itself for every segment of molecular dynamics, whatever the template says. The values of these tags in `inputs/INCAR` are therefore not the ones that were used.

| Tag | Value in the template | Value used |
|---|---|---|
| `NSW` | 200000 | 80 steps per segment. The gas exchange runs between two segments |
| `TEBEG`, `TEEND` | 873 | target temperature of the trajectory |
| `SIGMA` | 0.2064 | 8.6e-5 eV/K times the target temperature, 0.075 to 0.110 eV |
| `SMASS` | 0 | 0 (Nosé thermostat) |
| `POTIM` | 1 | adapted after every segment so that the electronic loop takes about `tgt_nelm` = 6 steps (`adj_potim = 1` in `job.in`). About 1.1 fs on average over the 16 trajectories |
| `BMIX` | 0.63 | adapted during the run (`adj_bmix = 1` in `job.in`) |
| `NBANDS` | not set | chosen by VASP for the first segment and then adjusted by the workflow |

Further settings that are not in the files:

- Plane-wave cutoff: no `ENCUT` is set, so VASP uses the largest value of the potentials, 400 eV.
- k-points: `kmesh = -1` in `job.in` selects one special k-point at (0.25, 0.25, 0.25).
- Potentials: PAW PBE, `O` (08Apr2002), `Zr_sv` (04Jan2005) and `C` (08Apr2002), in the order of the `POSCAR`. The `POTCAR` files are not included, because the VASP licence does not allow their distribution.
- Spin polarization is not used.

### Gas exchange

Every 80 steps the framework removes CO and CO2 from the cell and adds one O2 molecule when the oxygen in the gas falls below a threshold. `AtomicRadiusTol` scales the sum of two covalent radii to give the distance below which two atoms count as bonded. `O2Tol` is the threshold at the initial gas volume and `OSmoothing` the weight of the exponential smoothing of the O2 count. The procedure is described in the Methods of the paper.

## Analysis scripts

```
analysis/
  Paths.py             where the data are, shared by all scripts
  core/                events, carbon partitioning, oxide layering, oxygen supply, mass gain
  carbon_states/       the five carbon states, commitment and liberation rates
  gas_exchange/        validation of the gas exchange, figure of the protocol
  spin_polarisation/   spin-polarized reference calculations
  convergence/         calculations with a higher cutoff and denser k-point meshes
  lib/                 modules that the scripts import, not run by hand
```

### Requirements

Python 3.9 or later with `numpy`, `scipy`, `pandas`, `matplotlib` and `lifelines`. The two scripts that render atomic structures (`SnapshotsRender.py`, `MethodRender.py`) need the `ovito` Python package.

### Data

The scripts read the trajectory of each simulation (`<T>_<cell>.xyz`, for example `973_1.xyz`) and its gas-exchange log (`RateAnalysis_<T>_<cell>.csv`). Set the environment variable `ZRC_DATA_DIR` to the folder that holds these files, or put them in `analysis/data/`. `python analysis/Paths.py` reports whether they are found.

Every script is run from its own folder and writes its results to a folder beside itself. Later scripts read the results of earlier ones, so the scripts of each folder are listed in the order in which they run.

### core/

| Step | Script | What it produces | Used for |
|---|---|---|---|
| 1 | `GasSupplyAudit.py` | Oxygen pressure and flux of every trajectory, flux against time | Table of the simulation conditions, flux figure of the Supplementary Information |
| 2 | `Refresh140.py` | Events and product counts, density profiles, oxide layers, and the frame caches that the later scripts read | Table of the products, layering table |
| 3 | `RateAnalysisAudit.py` | Comparison of the gas-exchange logs with the trajectories | Audit of the logs in the Supplementary Information |
| 4 | `CommitmentCompetingRisks.py` | Commitment rates on the event table | Input of steps 6 and of `carbon_states/` |
| 5 | `SubsequentChainFate.py` | Fate of chain carbon, identities of the erroneously deleted atoms | Input of step 7 |
| 6 | `WeibullAudit.py` | One record per carbon atom, check of the Weibull fits | Input of step 8 and of `carbon_states/` |
| 7 | `PhiTimeseries.py` | Carbon product ratio and oxygen retention ratio against time | Ratios in the table of the products, time-series figure of the Supplementary Information |
| 8 | `IndependentTrajectoryReview.py` | Ledger of every carbon atom per trajectory | Input of `carbon_states/` |
| 9 | `BootstrapFigureRemake.py` | Bootstrap of the carbon partitioning | Figure of the carbon product ratio |
| 10 | `Window10Rerun.py` | Density profiles, layering with a shorter averaging window | Figure of the density profiles |
| 11 | `ExemplarProfilesFigure.py` | Layer detection on two surfaces | Figure in the Supplementary Information |
| 12 | `DiscussionSupportNotes.py` | Net mass gain, mass flows, net-mass activation energy with and without division by the flux | Figure of the mass flows |
| 13 | `RegenerateTrajTable.py` | Products, ratios and flux of every trajectory | Per-trajectory table of the Supplementary Information |
| 14 | `SnapshotsRender.py`, then `SnapshotsFigure.py` | Rendered structures at 140 ps | Figure of the snapshots |

`SnapshotsRender.py` reads the frames at 140 ps of two trajectories as `SnapshotsResults/Frame140_973_3.xyz` and `Frame140_1273_3.xyz`.

`Refresh140.py` builds its frame caches on the first run. A complete run takes more than half an hour. `python Refresh140.py --modules 1,2,3,4,6,7,9` runs the parts that the paper uses (events, partitioning, rates, profiles, chains, layering) in about ten minutes once the caches exist.

`Window10Rerun.py` draws the density profiles at the text width of the journal by default. `GasSupplyAudit.py --flux-figure-only` and `PhiTimeseries.py --si-figures-only` redraw their figures from the stored results.

### carbon_states/

| Step | Script | What it produces | Used for |
|---|---|---|---|
| 1 | `RawCoordinationPass.py`, `RawOriginalNeighbourPass.py` | Zr and C neighbour counts of every carbon atom at every MD step | Input of the next steps |
| 2 | `ForwardEventAnalyses.py` | Commitment rates and their Weibull shapes, bootstraps of the rates and activation energies, flux-normalized commitment energy, liberation rates, sensitivity to the state definitions | Commitment and liberation numbers in the text, survival tables of the Supplementary Information |
| 3 | `CarbonStatesFigure.py` | States, transitions and commitment rates. `--with-liberation` also plots the liberation rate | Figure of the carbon states |

These need steps 1 to 8 of `core/`.

### gas_exchange/

| Step | Script | What it produces | Used for |
|---|---|---|---|
| 1 | `BuildCache.py` | The trajectories as compact arrays (about 4 GB, in the system temporary folder, or in the folder named by `R27_CACHE`) | Input of the next steps |
| 2 | `GasExchangeAnalysis.py` | Every removal and addition followed frame by frame, detection repeated with shorter and longer bond cutoffs, kinetic temperatures | Input of the next steps |
| 3 | `Summarise.py` | Tables and figure of the removed molecules, of the oxygen supply and of the molecules removed at a later check | Validation of the gas exchange |
| 4 | `method_figure/MethodFigure.py` | Figure of the gas-exchange protocol. It calls `MethodRender.py` with the Python named by `OVITO_PYTHON` | Figure in the Methods |

### spin_polarisation/ and convergence/

| Script | What it produces | Used for |
|---|---|---|
| `spin_polarisation/SelectSnapshots.py` | The structures for the static calculations, as POSCAR files | Input of the calculations |
| `spin_polarisation/AnalyseSpinTest.py` | Energies, moments and forces with and without spin polarization | Figure in the Methods, figure and table in the Supplementary Information |
| `spin_polarisation/GasPhaseClusters.py` | Share of gas molecules in contact with another gas molecule | Text of the Supplementary Information |
| `convergence/AnalyseConvergence.py` | Forces and energy differences with a higher cutoff and denser k-point meshes | Figure and tables in the Supplementary Information |

`AnalyseSpinTest.py` and `AnalyseConvergence.py` read the energies, forces and magnetic moments of the static VASP calculations from `Results/results.json` in their folder. Both files are included, together with the structures of the calculations in `spin_polarisation/Structures/` (one folder per structure with its `POSCAR`, the starting magnetic moments and the atom groups). These two scripts therefore run without the trajectories.

### lib/

| Module | Content |
|---|---|
| `BootstrapCalculation.py` | Reading of the trajectories, bond criteria, tracking of atom identities, bootstrap of the carbon partitioning |
| `ChainTopology.py`, `ChainGeometryStratified.py` | Carbon chains |
| `OxygenLayeringAnalysis.py` | Detection of oxygen layers and carbide planes |
| `BoundaryConventions.py` | Conventions at the 140 ps horizon |
| `SurvivalAnalysis.py` | Rate and Weibull fits with censoring |
| `EaBootstrap.py` | Bootstrap of the activation energies |
| `SquareModel.py` | The carbon states from the two properties (Zr coordination, carbon bonding) |
| `SpinData.py` | Loading of the results of the static calculations |

### Notes

- `core/GasSupplyAudit.py` can use the post-processing module of the framework. Set `SGUSCHI_POSTPROCESSING` to the `src/postprocessing` folder of SGUSCHI. Without it the script uses its own implementation of the same calculation.
- `core/WeibullAudit.py` compares its fits with a table in the LaTeX source of the Supplementary Information when `ZRC_PAPER_DIR` points to the folder that holds `Supplementary/supplementary.tex`. Without it that comparison is skipped.
- The scripts are deterministic. The bootstraps use the seed 20260715.

## Licence

The contents of this repository are released under CC0 1.0 Universal (see `LICENSE`), like the SGUSCHI framework.
