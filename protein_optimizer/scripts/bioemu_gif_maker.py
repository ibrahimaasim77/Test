#!/usr/bin/env python3
"""
bioemu_gif_maker.py

Takes a directory of PDB files from BioEmu structural-ensemble output,
clusters the conformations into states, filters out structures too far
from any state centre, and writes an animated GIF per state.

Clustering logic mirrors ConformationalLandscapeScorer in scoring.py:
  - Feature = upper-triangle of Cα pairwise distance matrix (dRMSD fingerprint)
  - State prototypes chosen by greedy max-coverage selection
  - Outlier threshold = dRMSD > --rmsd_cutoff from the nearest prototype

Usage
-----
python scripts/bioemu_gif_maker.py --pdb_dir /path/to/bioemu_pdbs
python scripts/bioemu_gif_maker.py --pdb_dir /path/to/bioemu_pdbs \\
    --output_dir ./gifs --n_states 3 --rmsd_cutoff 4.0 --fps 4

Optional rotating-view mode (one GIF per state showing prototype spinning):
python scripts/bioemu_gif_maker.py --pdb_dir /path/to/bioemu_pdbs --rotate

Dependencies
------------
pip install biopython numpy matplotlib imageio Pillow
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports with friendly error messages
# ---------------------------------------------------------------------------

def _require_biopython():
    try:
        from Bio.PDB import PDBParser
        return PDBParser
    except ImportError:
        sys.exit("Missing dependency — install Biopython:  pip install biopython")


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / no display needed
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
        return plt
    except ImportError:
        sys.exit("Missing dependency — install matplotlib:  pip install matplotlib")


def _require_imageio():
    try:
        import imageio
        return imageio
    except ImportError:
        sys.exit("Missing dependency — install imageio and Pillow:  pip install imageio Pillow")


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------


def load_ca_coords(pdb_path: Path, PDBParser) -> Optional[np.ndarray]:
    """
    Parse Cα coordinates from a PDB file.

    Returns an (L, 3) float array, or None if parsing fails / too few atoms.
    Only the first MODEL is read, all chains are concatenated in order.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("s", str(pdb_path))
    except Exception as exc:
        logger.warning("Cannot parse %s: %s", pdb_path.name, exc)
        return None

    ca_coords: List[np.ndarray] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " " and "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())
        break  # first MODEL only

    if len(ca_coords) < 4:
        logger.warning("Too few Cα atoms (%d) in %s — skipping", len(ca_coords), pdb_path.name)
        return None

    return np.array(ca_coords, dtype=float)


# ---------------------------------------------------------------------------
# Feature extraction (dRMSD fingerprint)
# ---------------------------------------------------------------------------


def coords_to_fingerprint(ca_coords: np.ndarray) -> np.ndarray:
    """
    Upper-triangle of the Cα pairwise distance matrix, flattened to 1-D.

    This is the same representation used by ConformationalLandscapeScorer in
    scoring.py — a coordinate-frame-independent structural fingerprint.
    Two structures with similar topology will have a small dRMSD between their
    fingerprints regardless of orientation or translation.
    """
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    L = len(ca_coords)
    return dist_matrix[np.triu_indices(L, k=1)]


# ---------------------------------------------------------------------------
# State clustering
# ---------------------------------------------------------------------------


def select_prototypes(features: np.ndarray, n_states: int) -> List[int]:
    """
    Greedy max-coverage prototype selection.

    Picks the first prototype as index 0, then repeatedly selects the sample
    that is farthest (in dRMSD) from the already-selected set.  This is the
    same algorithm used in ConformationalLandscapeScorer._select_target_prototypes.

    Returns a list of integer indices into `features`.
    """
    n_samples = features.shape[0]
    n_states = min(n_states, n_samples)
    selected: List[int] = [0]

    while len(selected) < n_states:
        sel_feats = features[selected]
        # dRMSD: (n_samples, n_selected)
        deltas = features[:, None, :] - sel_feats[None, :, :]
        dists = np.sqrt(np.mean(deltas ** 2, axis=2))
        min_dists = np.min(dists, axis=1)
        next_idx = int(np.argmax(min_dists))
        if next_idx in selected:
            break
        selected.append(next_idx)

    return selected


def assign_to_states(
    features: np.ndarray,
    prototype_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each sample to its nearest prototype and return the distances.

    Returns
    -------
    assignments  : (N,) int array — state index for each sample
    min_dists    : (N,) float array — dRMSD to nearest prototype (Å)
    """
    deltas = features[:, None, :] - prototype_features[None, :, :]
    dists = np.sqrt(np.mean(deltas ** 2, axis=2))
    assignments = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(features)), assignments]
    return assignments, min_dists


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _axis_limits(all_coords: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute a consistent axis range across all structures so every frame uses
    the same scale, making the GIF easier to read.
    """
    all_centered = [c - c.mean(axis=0) for c in all_coords]
    all_vals = np.concatenate([c.ravel() for c in all_centered])
    limit = float(np.abs(all_vals).max()) * 1.15
    return (-limit, limit)


def render_frame(
    ca_coords: np.ndarray,
    title: str,
    axis_limits: Tuple[float, float],
    elev: float = 20.0,
    azim: float = 45.0,
    dpi: int = 100,
    fig_size: float = 6.0,
) -> np.ndarray:
    """
    Render a single Cα backbone trace to an (H, W, 3) uint8 RGB array.

    The backbone is coloured N→C terminus using a blue-to-red gradient.
    N-terminal Cα is marked with a filled blue circle, C-terminal with red.
    """
    plt = _require_matplotlib()

    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    centered = ca_coords - ca_coords.mean(axis=0)
    L = len(centered)
    colors = plt.cm.coolwarm(np.linspace(0, 1, L))

    for i in range(L - 1):
        ax.plot(
            centered[i : i + 2, 0],
            centered[i : i + 2, 1],
            centered[i : i + 2, 2],
            color=colors[i],
            linewidth=2.0,
            alpha=0.88,
        )

    ax.scatter(*centered[0], color="royalblue", s=60, zorder=5, label="N-term")
    ax.scatter(*centered[-1], color="firebrick", s=60, zorder=5, label="C-term")

    lim = axis_limits
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)
    ax.set_xlabel("X (Å)", fontsize=7)
    ax.set_ylabel("Y (Å)", fontsize=7)
    ax.set_zlabel("Z (Å)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, pad=4)
    ax.legend(loc="upper left", fontsize=6, markerscale=0.7)
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


def save_gif(frames: List[np.ndarray], path: Path, fps: float) -> None:
    """Write a list of RGB arrays to an animated GIF via imageio."""
    imageio = _require_imageio()
    duration_ms = int(1000.0 / fps)
    # imageio v3 / Pillow backend expects duration in milliseconds
    try:
        imageio.mimsave(str(path), frames, format="GIF", duration=duration_ms, loop=0)
    except TypeError:
        # Older imageio v2 API expects seconds
        imageio.mimsave(str(path), frames, duration=1.0 / fps, loop=0)
    logger.info("Saved  %s  (%d frames, %.1f fps)", path.name, len(frames), fps)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_assignment_report(
    output_dir: Path,
    prefix: str,
    paths: List[Path],
    assignments: np.ndarray,
    min_dists: np.ndarray,
    keep_mask: np.ndarray,
    rmsd_cutoff: float,
) -> None:
    """
    Write a CSV summarising which PDB landed in which state and why it was
    kept or filtered.
    """
    csv_path = output_dir / f"{prefix}assignments.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["pdb_file", "state", "drmsd_to_state", "kept", "rmsd_cutoff"],
        )
        writer.writeheader()
        for pdb, state, dist, kept in zip(paths, assignments, min_dists, keep_mask):
            writer.writerow(
                {
                    "pdb_file": pdb.name,
                    "state": int(state) + 1,
                    "drmsd_to_state": f"{dist:.4f}",
                    "kept": "yes" if kept else "no (outlier)",
                    "rmsd_cutoff": rmsd_cutoff,
                }
            )
    logger.info("Assignment report → %s", csv_path.name)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    pdb_dir: Path,
    output_dir: Path,
    n_states: int,
    rmsd_cutoff: float,
    fps: float,
    min_frames: int,
    rotate: bool,
    n_rotation_frames: int,
    dpi: int,
    fig_size: float,
) -> None:
    PDBParser = _require_biopython()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load PDB files ────────────────────────────────────────────────────
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        logger.error("No *.pdb files found in %s", pdb_dir)
        sys.exit(1)
    logger.info("Found %d PDB files in %s", len(pdb_files), pdb_dir)

    structures: List[Tuple[Path, np.ndarray]] = []
    for pdb in pdb_files:
        coords = load_ca_coords(pdb, PDBParser)
        if coords is not None:
            structures.append((pdb, coords))

    if not structures:
        logger.error("Could not parse Cα coordinates from any PDB file")
        sys.exit(1)

    logger.info("Successfully parsed %d / %d PDB files", len(structures), len(pdb_files))

    # ── Group by residue count ────────────────────────────────────────────
    # BioEmu output for one sequence will all have the same length.
    # Multiple sequences in the same directory are handled separately.
    length_groups: Dict[int, List[Tuple[Path, np.ndarray]]] = {}
    for pdb, coords in structures:
        length_groups.setdefault(len(coords), []).append((pdb, coords))

    if len(length_groups) > 1:
        logger.warning(
            "Multiple residue lengths detected: %s — each group is processed independently.",
            {L: len(g) for L, g in sorted(length_groups.items())},
        )

    # ── Per-length-group processing ───────────────────────────────────────
    total_gifs = 0
    total_filtered = 0

    for group_L, group in sorted(length_groups.items()):
        group_paths = [p for p, _ in group]
        group_coords = [c for _, c in group]
        n = len(group)
        prefix = f"L{group_L}_" if len(length_groups) > 1 else ""

        logger.info(
            "── Group L=%d residues | %d structures ──", group_L, n
        )

        # Compute fingerprints
        features = np.vstack([coords_to_fingerprint(c) for c in group_coords])

        # Select state prototypes
        actual_states = min(n_states, n)
        proto_indices = select_prototypes(features, actual_states)
        proto_features = features[proto_indices]

        logger.info(
            "  State prototypes (indices): %s",
            [group_paths[i].name for i in proto_indices],
        )

        # Assign all structures to nearest state and compute dRMSD
        assignments, min_dists = assign_to_states(features, proto_features)

        # Filter outliers
        keep_mask = min_dists <= rmsd_cutoff
        n_filtered = int(np.sum(~keep_mask))
        total_filtered += n_filtered
        n_kept = int(np.sum(keep_mask))

        if n_filtered:
            logger.info(
                "  Filtered %d outlier(s) with dRMSD > %.2f Å  (%d kept)",
                n_filtered, rmsd_cutoff, n_kept,
            )
        else:
            logger.info(
                "  No outliers — all %d structures within dRMSD %.2f Å", n, rmsd_cutoff
            )

        # Save assignment report
        write_assignment_report(
            output_dir, prefix, group_paths,
            assignments, min_dists, keep_mask, rmsd_cutoff,
        )

        # Consistent axis limits across all kept structures
        kept_coords = [group_coords[i] for i in range(n) if keep_mask[i]]
        if not kept_coords:
            logger.warning("  No structures kept after filtering — skipping GIF creation")
            continue
        axis_limits = _axis_limits(kept_coords)

        # ── Per-state GIF ─────────────────────────────────────────────────
        for state_idx, proto_idx in enumerate(proto_indices):
            state_mask = (assignments == state_idx) & keep_mask
            state_indices = [i for i in range(n) if state_mask[i]]

            if len(state_indices) < min_frames:
                logger.warning(
                    "  State %d: only %d frame(s) after filtering (min=%d) — skipping",
                    state_idx + 1, len(state_indices), min_frames,
                )
                continue

            state_paths = [group_paths[i] for i in state_indices]
            state_coords = [group_coords[i] for i in state_indices]
            state_dists = min_dists[state_indices]

            # Sort closest-to-prototype first so the GIF starts at the canonical form
            order = np.argsort(state_dists)
            state_paths = [state_paths[i] for i in order]
            state_coords = [state_coords[i] for i in order]
            state_dists = state_dists[order]

            logger.info(
                "  State %d: %d structures  dRMSD %.2f – %.2f Å",
                state_idx + 1,
                len(state_paths),
                float(state_dists[0]),
                float(state_dists[-1]),
            )

            frames: List[np.ndarray] = []

            if rotate:
                # Rotating view of the prototype structure
                proto_coords = group_coords[proto_idx]
                for frame_i, azim in enumerate(
                    np.linspace(0, 360, n_rotation_frames, endpoint=False)
                ):
                    title = (
                        f"State {state_idx + 1}/{actual_states}  |  "
                        f"Prototype: {group_paths[proto_idx].name}\n"
                        f"L={group_L} residues  |  frame {frame_i + 1}/{n_rotation_frames}"
                    )
                    frame = render_frame(
                        proto_coords, title, axis_limits,
                        azim=float(azim), dpi=dpi, fig_size=fig_size,
                    )
                    frames.append(frame)
                    if (frame_i + 1) % 12 == 0:
                        logger.debug(
                            "    State %d rotate: %d/%d frames rendered",
                            state_idx + 1, frame_i + 1, n_rotation_frames,
                        )
            else:
                # One frame per structure
                n_frames = len(state_paths)
                for frame_i, (pdb_path, coords) in enumerate(
                    zip(state_paths, state_coords)
                ):
                    title = (
                        f"State {state_idx + 1}/{actual_states}  |  {pdb_path.name}\n"
                        f"dRMSD to state = {state_dists[frame_i]:.2f} Å  |  "
                        f"frame {frame_i + 1}/{n_frames}"
                    )
                    frame = render_frame(
                        coords, title, axis_limits, dpi=dpi, fig_size=fig_size
                    )
                    frames.append(frame)
                    if (frame_i + 1) % 10 == 0:
                        logger.debug(
                            "    State %d: %d/%d frames rendered",
                            state_idx + 1, frame_i + 1, n_frames,
                        )

            gif_path = output_dir / f"{prefix}state_{state_idx + 1:02d}.gif"
            save_gif(frames, gif_path, fps=fps)
            total_gifs += 1

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 54)
    print("  BioEmu GIF Maker — Complete")
    print("=" * 54)
    print(f"  Input PDB dir      : {pdb_dir.resolve()}")
    print(f"  Structures parsed  : {len(structures)}")
    print(f"  Outliers filtered  : {total_filtered}  (dRMSD > {rmsd_cutoff} Å)")
    print(f"  GIFs created       : {total_gifs}")
    print(f"  Output dir         : {output_dir.resolve()}")
    print("=" * 54)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-state animated GIFs from BioEmu PDB output, "
            "filtering out structures too far from each conformational state."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--pdb_dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing BioEmu *.pdb files",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./bioemu_gifs"),
        metavar="DIR",
        help="Directory to write GIF files and the assignment report CSV",
    )

    # Clustering
    parser.add_argument(
        "--n_states",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Number of conformational states to cluster into using greedy "
            "max-coverage prototype selection"
        ),
    )
    parser.add_argument(
        "--rmsd_cutoff",
        type=float,
        default=4.0,
        metavar="Å",
        help=(
            "dRMSD threshold (Å).  Structures whose distance-matrix fingerprint "
            "dRMSD to the nearest state prototype exceeds this value are excluded "
            "from the GIF as outliers."
        ),
    )

    # GIF options
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        metavar="FPS",
        help="Frames per second of the output GIF",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=2,
        metavar="N",
        help="Minimum number of frames a state must have to write a GIF",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help=(
            "Instead of one-frame-per-structure, create a 360° rotating view "
            "of the state prototype"
        ),
    )
    parser.add_argument(
        "--n_rotation_frames",
        type=int,
        default=36,
        metavar="N",
        help="Number of rotation frames per GIF (only used with --rotate)",
    )

    # Rendering
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI of each rendered frame",
    )
    parser.add_argument(
        "--fig_size",
        type=float,
        default=6.0,
        metavar="INCHES",
        help="Figure width and height in inches (square)",
    )

    # Logging
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s %(name)s  %(message)s",
    )

    run(
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        n_states=args.n_states,
        rmsd_cutoff=args.rmsd_cutoff,
        fps=args.fps,
        min_frames=args.min_frames,
        rotate=args.rotate,
        n_rotation_frames=args.n_rotation_frames,
        dpi=args.dpi,
        fig_size=args.fig_size,
    )


if __name__ == "__main__":
    main()
