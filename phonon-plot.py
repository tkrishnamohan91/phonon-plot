#!/usr/bin/env python
"""
phonon-plot.py
Compare phonon band structures from multiple phonopy runs on a single plot.

Uses sumo (bradcrack) to determine the k-path from the first run and applies
the same path to all runs, ensuring a consistent x-axis for comparison.

Requirements: sumo, pymatgen, phonopy, matplotlib
Run with: /Users/krishna/anaconda3/bin/python phonon-plot.py
"""

import itertools
import json
import os
import sys
import tempfile
import warnings

import matplotlib
import numpy as np

warnings.filterwarnings("ignore")

# Named colour presets for quick selection
COLOUR_PRESETS = {
    "orangered": "Orangered",
    "indigo":    "Indigo",
    "teal":      "Teal",
    "steelblue": "SteelBlue",
    "crimson":   "Crimson",
    "darkorange":"DarkOrange",
    "purple":    "Purple",
    "grey":      "Grey",
    "black":     "black",
}
DEFAULT_COLOURS = ["Orangered", "Indigo", "Teal", "Crimson", "DarkOrange", "Purple"]

from matplotlib.lines import Line2D
from phonopy.units import AMU, EV, Angstrom, PlanckConstant, THzToCm, VaspToTHz, pi, sqrt
from pymatgen.io.phonopy import get_ph_bs_symm_line
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.plotter import PhononBSPlotter as PmgPhononBSPlotter
from sumo.phonon.phonopy import load_phonopy
from sumo.plotting.phonon_bs_plotter import SPhononBSPlotter
from sumo.symmetry.kpoints import get_path_data

# Frequency conversion factors (matching sumo CLI conventions)
VaspToCm = VaspToTHz * THzToCm
VaspToEv = sqrt(EV / AMU) / Angstrom / (2 * pi) * PlanckConstant

UNIT_FACTORS = {
    "thz":  VaspToTHz,
    "cm-1": VaspToCm,
    "ev":   VaspToEv,
    "mev":  VaspToEv * 1000,
}


def get_supercell_dim(run_dir):
    """
    Return (dim_matrix 3x3, unitcell Structure) for a phonopy run directory.
    Reads SPOSCAR/POSCAR by default; falls back to phonopy.yaml dim setting.
    """
    poscar_path = os.path.join(run_dir, "POSCAR")
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"POSCAR not found in {run_dir}")

    poscar = Poscar.from_file(poscar_path)

    sposcar_path = os.path.join(run_dir, "SPOSCAR")
    if os.path.exists(sposcar_path):
        sposcar = Poscar.from_file(sposcar_path)
        dim = sposcar.structure.lattice.matrix @ poscar.structure.lattice.inv_matrix
        return np.around(dim, 5), poscar.structure

    # Fall back: parse dim from phonopy.yaml header
    phonopy_yaml = os.path.join(run_dir, "phonopy.yaml")
    if os.path.exists(phonopy_yaml):
        with open(phonopy_yaml) as f:
            for line in f:
                if "dim:" in line and '"' in line:
                    parts = line.split('"')[1].split()
                    if len(parts) == 3:
                        return np.diag(list(map(float, parts))), poscar.structure

    raise FileNotFoundError(
        f"Cannot determine supercell dimensions for '{run_dir}'.\n"
        "  Provide a SPOSCAR file or ensure phonopy.yaml contains a 'dim' entry."
    )


def compute_band_structure(run_dir, kpoints, labels, kpath, factor):
    """
    Load FORCE_SETS from run_dir, compute the phonon band structure along the
    given kpoints, and return a PhononBandStructureSymmLine object.
    """
    force_sets = os.path.join(run_dir, "FORCE_SETS")
    if not os.path.exists(force_sets):
        raise FileNotFoundError(f"FORCE_SETS not found in {run_dir}")

    dim, structure = get_supercell_dim(run_dir)

    phonon = load_phonopy(
        force_sets,
        structure,
        dim,
        factor=factor,
        symmetrise=True,
    )

    phonon.run_band_structure(kpoints, with_eigenvectors=False, labels=labels)

    # Write intermediate yaml, read back as pymatgen band structure object
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
    tmp.close()
    try:
        phonon.band_structure.write_yaml(filename=tmp.name)
        bs = get_ph_bs_symm_line(tmp.name, has_nac=False, labels_dict=kpath.kpoints)
    finally:
        os.unlink(tmp.name)

    return bs


def prompt(text, default=None):
    """Prompt with an optional default shown in brackets."""
    if default is not None:
        result = input(f"{text} [{default}]: ").strip()
        return result if result else default
    return input(f"{text}: ").strip()


def save_dos_comparison(dos_arrays, run_colours, run_labels, units_str, out_file):
    """
    Save a standalone DOS comparison figure (frequency vs DOS, all runs overlaid).
    dos_arrays: list parallel to run_colours/run_labels; None entries are skipped.
    """
    import matplotlib.pyplot as mplt

    fig, ax = mplt.subplots(figsize=(6, 4.5))

    for dos_arr, colour, label in zip(dos_arrays, run_colours, run_labels):
        if dos_arr is not None:
            ax.plot(dos_arr[:, 0], dos_arr[:, 1], color=colour, lw=1.0, label=label)

    ax.axvline(0, color="grey", ls="--", lw=0.5)
    ax.set_xlabel(f"Frequency ({units_str})", fontsize=14)
    ax.set_ylabel("DOS", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    fig.tight_layout()

    fmt = out_file.rsplit(".", 1)[-1] if "." in out_file else "pdf"
    fig.savefig(out_file, format=fmt, dpi=200, bbox_inches="tight")
    mplt.close(fig)


def main():
    print("=== Phonon Band Structure Comparison ===\n")

    # Collect run directories and labels
    try:
        n_runs = int(prompt("Number of phonopy runs to compare"))
    except ValueError:
        print("Error: enter an integer.")
        sys.exit(1)

    if n_runs < 1:
        print("Error: need at least one run.")
        sys.exit(1)

    colour_hint = ", ".join(DEFAULT_COLOURS[:6])
    print(f"  Colour suggestions: {colour_hint}")
    print("  (Any matplotlib colour name or hex code, e.g. '#1f77b4')\n")

    run_dirs, run_labels, run_colours = [], [], []
    for i in range(n_runs):
        default_colour = DEFAULT_COLOURS[i % len(DEFAULT_COLOURS)]
        print(f"\n  Run {i + 1} of {n_runs}")
        d      = prompt("    Directory path")
        lbl    = prompt("    Label (e.g. VASP, DFT-FE-0.75)", default=f"Run {i + 1}")
        colour = prompt("    Colour", default=default_colour)
        # Accept preset aliases case-insensitively
        colour = COLOUR_PRESETS.get(colour.lower(), colour)
        if not os.path.isdir(d):
            print(f"  Warning: '{d}' is not a directory.")
        run_dirs.append(d)
        run_labels.append(lbl)
        run_colours.append(colour)

    print()
    line_density = int(prompt("K-point line density", default="60"))
    units_str    = prompt("Frequency units  (THz / cm-1 / eV / meV)", default="THz")
    want_dos     = prompt("Include DOS comparison panel? (yes/no)", default="yes").lower() in ("yes", "y")
    out_file     = prompt("Output filename", default="phonon_comparison.pdf")

    factor = UNIT_FACTORS.get(units_str.lower(), VaspToTHz)

    # Determine k-path from the first run; reuse it for all others
    print(f"\nDetermining k-path from: {run_dirs[0]}")
    _, ref_structure = get_supercell_dim(run_dirs[0])
    kpath, kpoints, labels = get_path_data(
        ref_structure,
        mode="bradcrack",
        line_density=line_density,
        phonopy=True,
    )
    path_str = " → ".join(kpath.kpoints.keys())
    print(f"  k-path: {path_str}")

    # Load DOS data (total_dos.dat, two columns: frequency, DOS)
    dos_arrays = []
    if want_dos:
        print()
        for run_dir, label in zip(run_dirs, run_labels):
            dos_file = os.path.join(run_dir, "total_dos.dat")
            if os.path.exists(dos_file):
                dos_arrays.append(np.loadtxt(dos_file, comments="#"))
                print(f"  DOS loaded: {label}")
            else:
                dos_arrays.append(None)
                print(f"  DOS not found for '{label}' ({dos_file}) — skipping")

    # Compute band structure for each run along the shared k-path
    band_structures = []
    for run_dir, label in zip(run_dirs, run_labels):
        print(f"\nComputing: {label}  ({run_dir})")
        bs = compute_band_structure(run_dir, kpoints, labels, kpath, factor)
        band_structures.append(bs)
        print(f"  {bs.nb_bands} branches, {len(bs.distance)} q-points")

    # Build comparison plot
    print("\nGenerating plot ...")

    # sumo sets up the axes (ticks, labels, zero line) for the first run only.
    # We pass no from_json and no legend so sumo doesn't draw a legend itself.
    plotter = SPhononBSPlotter(band_structures[0])
    plt_obj = plotter.get_plot(
        units=units_str,
        color=run_colours[0],
        width=6.0,
        height=6.0,
    )
    ax = plt_obj.gca()

    # Plot each additional run directly on the same axes.
    # Adjust lattice_rec to match the reference so x-axis distances are consistent.
    ref_lattice_rec = json.loads(band_structures[0].lattice_rec.to_json())
    for bs, colour in zip(band_structures[1:], run_colours[1:]):
        bs_dict = json.loads(bs.to_json())
        bs_dict["lattice_rec"] = ref_lattice_rec
        bs_adj = PhononBandStructureSymmLine.from_dict(bs_dict)

        data = PmgPhononBSPlotter(bs_adj).bs_plot_data()
        for nd, nb in itertools.product(
            range(len(data["distances"])), range(bs_adj.nb_bands)
        ):
            ax.plot(data["distances"][nd], data["frequency"][nd][nb],
                    ls="-", c=colour, lw=1.0, zorder=0.5)

    # Add legend manually with one coloured line per run
    ax.legend(
        [Line2D([0], [0], color=c, lw=1.5) for c in run_colours],
        run_labels,
        fontsize=12,
    )

    fmt = out_file.rsplit(".", 1)[-1] if "." in out_file else "pdf"
    plt_obj.savefig(out_file, format=fmt, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_file}")

    # Save DOS comparison as a separate figure
    if want_dos and any(d is not None for d in dos_arrays):
        stem = out_file.rsplit(".", 1)[0] if "." in out_file else out_file
        dos_out = f"{stem}_dos.{fmt}"
        save_dos_comparison(dos_arrays, run_colours, run_labels, units_str, dos_out)
        print(f"Saved: {dos_out}")


if __name__ == "__main__":
    main()
