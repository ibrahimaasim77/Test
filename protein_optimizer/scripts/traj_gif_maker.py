#!/usr/bin/env python3
"""
traj_gif_maker.py

Reads a PDB topology + trajectory file (XTC, DCD, TRR, …) and creates an
animated GIF of the Cα backbone across frames.

Two modes
---------
1. Single GIF (default)
   All frames are superposed onto the first frame and rendered into one GIF.
   Optional RMSD cutoff (--rmsd_cutoff) filters frames too far from frame 0.

2. Per-state GIFs  (--n_states N)
   Frames are clustered into N conformational states using the same greedy
   max-coverage / dRMSD fingerprint approach as ConformationalLandscapeScorer
   (scoring.py).  Frames farther than --rmsd_cutoff from their state prototype
   are discarded.  One GIF is written per state.

Usage
-----
# Quick single GIF — every frame, no filtering
python scripts/traj_gif_maker.py --pdb "PDB Data/SB40.pdb" --traj "PDB Data/SB40.xtc"

# Stride + RMSD filter
python scripts/traj_gif_maker.py --pdb "PDB Data/SB40.pdb" --traj "PDB Data/SB40.xtc" \\
    --stride 2 --rmsd_cutoff 3.0 --fps 8

# State-based clustering (3 states, filter outliers)
python scripts/traj_gif_maker.py --pdb "PDB Data/SB40.pdb" --traj "PDB Data/SB40.xtc" \\
    --n_states 3 --rmsd_cutoff 4.0

Renderers
---------
pymol (default)  — cartoon with secondary-structure coloring (helix / sheet / loop)
                   pip install pymol-open-source
matplotlib       — simple Cα backbone trace coloured N→C
                   pip install MDAnalysis numpy matplotlib imageio Pillow

In PyMOL mode secondary structure is recomputed per-frame via dss, so helices
and sheets animate as the protein moves through the trajectory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _require_mda():
    try:
        import MDAnalysis as mda
        return mda
    except ImportError:
        sys.exit(
            "Missing dependency — install MDAnalysis:\n"
            "  pip install MDAnalysis"
        )


def _require_mda_align():
    try:
        from MDAnalysis.analysis import align, rms
        return align, rms
    except ImportError:
        sys.exit("MDAnalysis analysis module not found — reinstall MDAnalysis")


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        return plt
    except ImportError:
        sys.exit("Missing dependency — install matplotlib:  pip install matplotlib")


def _require_imageio():
    try:
        import imageio
        return imageio
    except ImportError:
        sys.exit("Missing dependency:  pip install imageio Pillow")


def _require_pymol():
    try:
        import pymol2
        return pymol2
    except ImportError:
        sys.exit(
            "PyMOL is not installed.  Install the open-source build:\n"
            "  pip install pymol-open-source\n"
            "or via conda:\n"
            "  conda install -c conda-forge pymol-open-source"
        )


# ---------------------------------------------------------------------------
# Secondary structure via MDTraj DSSP
# ---------------------------------------------------------------------------

def compute_ss_mdtraj(
    pdb_path: Path,
    traj_path: Path,
    stride: int,
) -> Optional[np.ndarray]:
    """
    Compute per-residue DSSP secondary structure for all strided frames.

    Uses MDTraj's implementation of the DSSP algorithm (Kabsch & Sander 1983).

    Returns
    -------
    (n_frames, n_residues) array of str: 'H' (helix), 'E' (strand), 'C' (coil).
    Returns None if MDTraj is unavailable or the computation fails.
    """
    try:
        import mdtraj as md
    except ImportError:
        logger.warning("mdtraj not installed — SS coloring disabled  (pip install mdtraj)")
        return None

    try:
        logger.info("Loading trajectory with MDTraj for DSSP computation…")
        traj = md.load(str(traj_path), top=str(pdb_path))
        traj_s = traj[::stride]
        dssp = md.compute_dssp(traj_s, simplified=True)  # (n_frames, n_residues): H / E / C / NA
        logger.info("DSSP complete: %d frames × %d residues", *dssp.shape)
        return dssp
    except Exception as exc:
        logger.warning("DSSP computation failed (%s) — falling back to N→C coloring", exc)
        return None


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_ca_trajectory(
    pdb_path: Path,
    traj_path: Path,
    stride: int,
    align_frames: bool,
) -> Tuple[np.ndarray, int]:
    """
    Load Cα coordinates for every `stride`-th frame.

    Returns
    -------
    coords   : (n_frames, L, 3) float array
    n_atoms  : number of Cα atoms (= number of residues)
    """
    mda = _require_mda()

    logger.info("Loading topology: %s", pdb_path.name)
    logger.info("Loading trajectory: %s  (stride=%d)", traj_path.name, stride)

    u = mda.Universe(str(pdb_path), str(traj_path))
    ca = u.select_atoms("name CA")
    n_ca = len(ca)

    if n_ca == 0:
        sys.exit("No Cα atoms found — check PDB atom naming")

    logger.info("Cα atoms: %d  |  Trajectory frames: %d", n_ca, u.trajectory.n_frames)

    # Superpose all frames onto frame 0 (in-memory) for stable visualisation
    if align_frames:
        logger.info("Aligning trajectory to frame 0 (Cα backbone)…")
        align_mod, _ = _require_mda_align()
        try:
            align_mod.AlignTraj(
                u, u,
                select="name CA",
                in_memory=True,
            ).run()
            logger.info("Alignment complete")
        except Exception as exc:
            logger.warning("Alignment failed (%s) — proceeding unaligned", exc)

    frames: List[np.ndarray] = []
    for ts in u.trajectory[::stride]:
        frames.append(ca.positions.copy())

    coords = np.array(frames, dtype=float)   # (n_frames, L, 3)
    logger.info("Loaded %d frames  (L=%d residues)", len(coords), n_ca)
    return coords, n_ca


# ---------------------------------------------------------------------------
# dRMSD fingerprint & clustering (mirrors ConformationalLandscapeScorer)
# ---------------------------------------------------------------------------

def coords_to_fingerprint(ca_coords: np.ndarray) -> np.ndarray:
    """Upper-triangle of pairwise Cα distance matrix → 1-D fingerprint."""
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    L = len(ca_coords)
    return dist[np.triu_indices(L, k=1)]


def _pairwise_drmsd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    deltas = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.mean(deltas ** 2, axis=2))


def select_prototypes(features: np.ndarray, n_states: int) -> List[int]:
    """Greedy max-coverage prototype selection (same as scoring.py)."""
    n = features.shape[0]
    n_states = min(n_states, n)
    selected: List[int] = [0]
    while len(selected) < n_states:
        dists = _pairwise_drmsd(features, features[selected])
        min_d = np.min(dists, axis=1)
        nxt = int(np.argmax(min_d))
        if nxt in selected:
            break
        selected.append(nxt)
    return selected


def assign_to_states(
    features: np.ndarray,
    proto_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dists = _pairwise_drmsd(features, proto_features)
    assignments = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(features)), assignments]
    return assignments, min_dists


# ---------------------------------------------------------------------------
# Per-frame RMSD filtering (single-GIF mode)
# ---------------------------------------------------------------------------

def frame_rmsd_to_ref(coords: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """
    Compute per-frame Cα RMSD to a reference frame.

    Returns (n_frames,) float array.
    """
    ref = coords[ref_idx]
    diff = coords - ref[None, :, :]
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=-1), axis=-1))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# SS type → (color, linewidth, alpha)
_SS_STYLE: dict = {
    "H":  ("firebrick",   3.5, 0.95),   # alpha helix
    "E":  ("goldenrod",   2.5, 0.95),   # beta strand
    "C":  ("steelblue",   1.5, 0.75),   # coil / loop
    "NA": ("gray",        1.2, 0.60),   # not assigned
}


def _axis_limits(all_coords: np.ndarray) -> Tuple[float, float]:
    """Consistent axis range centred on the mean position."""
    centred = all_coords - all_coords.mean(axis=(0, 1), keepdims=True)
    limit = float(np.abs(centred).max()) * 1.15
    return (-limit, limit)


def render_frame(
    ca_coords: np.ndarray,
    title: str,
    axis_limits: Tuple[float, float],
    ss_labels: Optional[np.ndarray] = None,
    elev: float = 20.0,
    azim: float = 45.0,
    dpi: int = 100,
    fig_size: float = 6.0,
) -> np.ndarray:
    """
    Render one Cα backbone trace → (H, W, 3) uint8 RGB.

    When ss_labels is provided (one DSSP code per residue: H / E / C),
    each segment is coloured by secondary structure type:
        Helix  (H) — firebrick red,   thick line
        Strand (E) — goldenrod,        medium line
        Coil   (C) — steelblue,        thin line
    """
    plt = _require_matplotlib()

    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    centred = ca_coords - ca_coords.mean(axis=0)
    L = len(centred)

    use_ss = ss_labels is not None and len(ss_labels) == L

    if use_ss:
        # Draw each bond segment coloured by its residue's SS type
        for i in range(L - 1):
            ss = str(ss_labels[i])
            color, lw, alpha = _SS_STYLE.get(ss, _SS_STYLE["NA"])
            ax.plot(
                centred[i : i + 2, 0],
                centred[i : i + 2, 1],
                centred[i : i + 2, 2],
                color=color, linewidth=lw, alpha=alpha, solid_capstyle="round",
            )
        # N / C terminal markers
        ax.scatter(*centred[0],  color="black",   s=40, zorder=6, marker="o")
        ax.scatter(*centred[-1], color="black",   s=40, zorder=6, marker="s")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="firebrick", label="Helix (H)"),
            Patch(facecolor="goldenrod", label="Strand (E)"),
            Patch(facecolor="steelblue", label="Coil (C)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=6, framealpha=0.7)

    else:
        # Fallback: N→C coolwarm gradient
        colors = plt.cm.coolwarm(np.linspace(0, 1, L))
        for i in range(L - 1):
            ax.plot(
                centred[i : i + 2, 0],
                centred[i : i + 2, 1],
                centred[i : i + 2, 2],
                color=colors[i], linewidth=2.0, alpha=0.88,
            )
        ax.scatter(*centred[0],  color="royalblue", s=55, zorder=5, label="N")
        ax.scatter(*centred[-1], color="firebrick", s=55, zorder=5, label="C")
        ax.legend(loc="upper left", fontsize=6, markerscale=0.7)

    lim = axis_limits
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.set_xlabel("X (Å)", fontsize=7)
    ax.set_ylabel("Y (Å)", fontsize=7)
    ax.set_zlabel("Z (Å)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, pad=4)
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# PyMOL renderer
# ---------------------------------------------------------------------------

def _add_title_overlay(img_arr: np.ndarray, title: str, font_size: int = 15) -> np.ndarray:
    """Stamp a multi-line title onto a (H, W, 3) uint8 image using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img_arr

    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    lines = title.split("\n")
    y = 6
    for line in lines:
        # thin white halo for legibility on any background
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((8 + dx, y + dy), line, fill=(255, 255, 255), font=font)
        draw.text((8, y), line, fill=(20, 20, 20), font=font)
        y += font_size + 3

    return np.array(img)


def render_frames_pymol(
    pdb_path: Path,
    traj_path: Path,
    stride: int,
    keep_indices: List[int],
    frame_labels: List[str],
    width: int = 600,
    height: int = 600,
    helix_color: str = "firebrick",
    sheet_color: str = "wheat",
    loop_color: str = "slate",
    bg_color: str = "white",
) -> List[np.ndarray]:
    """
    Render kept trajectory frames using PyMOL cartoon + secondary-structure coloring.

    PyMOL's dss algorithm recomputes helix / sheet / loop assignments for each
    frame from actual atomic coordinates, so the coloring changes dynamically.

    Parameters
    ----------
    keep_indices  : 0-based indices into the strided frame list.
                    PyMOL state = keep_index + 1.
    frame_labels  : title strings (one per kept frame) overlaid with PIL.
    """
    pymol2 = _require_pymol()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("Missing dependency — install Pillow:  pip install Pillow")

    import tempfile

    frames_rgb: List[np.ndarray] = []
    keep_set = set(keep_indices)
    label_iter = iter(frame_labels)

    with pymol2.PyMOL() as p:
        logger.info("PyMOL: loading %s", pdb_path.name)
        p.cmd.load(str(pdb_path), "mol", state=1)

        logger.info("PyMOL: loading trajectory %s  (interval=%d)", traj_path.name, stride)
        p.cmd.load_traj(str(traj_path), "mol", interval=stride)

        n_pymol_states = p.cmd.count_states("mol")
        logger.info("PyMOL: %d states loaded", n_pymol_states)

        # Establish a fixed camera from the first state
        p.cmd.frame(1)
        p.cmd.orient("mol")
        p.cmd.zoom("mol", buffer=3)

        # Rendering settings
        p.cmd.bg_color(bg_color)
        p.cmd.set("cartoon_fancy_helices", 1)
        p.cmd.set("cartoon_smooth_loops", 1)
        p.cmd.set("ray_shadow", 0)
        p.cmd.set("ambient", 0.5)
        p.cmd.set("specular", 0.15)

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = str(Path(tmpdir) / "frame.png")

            for state in range(1, n_pymol_states + 1):
                frame_idx = state - 1  # 0-based into strided list
                if frame_idx not in keep_set:
                    continue

                p.cmd.frame(state)

                # Recompute secondary structure from this frame's coordinates
                p.cmd.dss("mol", state=state)

                # Apply representation and coloring
                p.cmd.show_as("cartoon", "mol")
                p.cmd.color(helix_color, "mol and ss h")
                p.cmd.color(sheet_color, "mol and ss s")
                p.cmd.color(loop_color,  "mol and ss l+")

                p.cmd.png(png_path, width=width, height=height, ray=0, quiet=1)

                img_arr = np.array(Image.open(png_path).convert("RGB"))
                img_arr = _add_title_overlay(img_arr, next(label_iter, ""))
                frames_rgb.append(img_arr)

                n_done = len(frames_rgb)
                if n_done % 20 == 0:
                    logger.info("  PyMOL: %d / %d frames rendered", n_done, len(keep_indices))

    logger.info("PyMOL rendering complete — %d frames", len(frames_rgb))
    return frames_rgb


# ---------------------------------------------------------------------------
# GIF assembly
# ---------------------------------------------------------------------------

def save_gif(frames: List[np.ndarray], path: Path, fps: float) -> None:
    imageio = _require_imageio()
    duration_ms = int(1000.0 / fps)
    try:
        imageio.mimsave(str(path), frames, format="GIF", duration=duration_ms, loop=0)
    except TypeError:
        imageio.mimsave(str(path), frames, duration=1.0 / fps, loop=0)
    logger.info("Saved  %s  (%d frames @ %.1f fps)", path.name, len(frames), fps)


# ---------------------------------------------------------------------------
# Report CSV
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    filename: str,
    frame_indices: List[int],
    assignments: np.ndarray,
    dists: np.ndarray,
    keep_mask: np.ndarray,
    cutoff: float,
) -> None:
    p = output_dir / filename
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "state", "distance", "kept", "cutoff"])
        w.writeheader()
        for fi, state, dist, kept in zip(frame_indices, assignments, dists, keep_mask):
            w.writerow({
                "frame": fi,
                "state": int(state) + 1,
                "distance": f"{dist:.4f}",
                "kept": "yes" if kept else "no (outlier)",
                "cutoff": cutoff,
            })
    logger.info("Report → %s", p.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    pdb_path: Path,
    traj_path: Path,
    output_dir: Path,
    stride: int,
    align_frames: bool,
    n_states: Optional[int],
    rmsd_cutoff: Optional[float],
    fps: float,
    min_frames: int,
    renderer: str,
    width: int,
    height: int,
    dpi: int,
    fig_size: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Cα trajectory (used for all filtering / clustering math) ──────
    coords, n_ca = load_ca_trajectory(pdb_path, traj_path, stride, align_frames)
    n_frames_total = len(coords)
    frame_indices = list(range(0, n_frames_total * stride, stride))
    axis_limits = _axis_limits(coords)
    stem = pdb_path.stem

    # ── Pre-compute DSSP for matplotlib renderer ───────────────────────────
    ss_all: Optional[np.ndarray] = None
    if renderer == "matplotlib":
        ss_all = compute_ss_mdtraj(pdb_path, traj_path, stride)
        if ss_all is not None and ss_all.shape[0] != n_frames_total:
            logger.warning(
                "DSSP frame count (%d) ≠ MDAnalysis frame count (%d) — SS disabled",
                ss_all.shape[0], n_frames_total,
            )
            ss_all = None

    # ── Renderer dispatch helper ───────────────────────────────────────────
    def _render_and_save(kept: List[int], labels: List[str], gif_name: str) -> None:
        if renderer == "pymol":
            frames_rgb = render_frames_pymol(
                pdb_path, traj_path, stride,
                kept, labels,
                width=width, height=height,
            )
        else:
            frames_rgb = []
            for render_i, fi in enumerate(kept):
                ss_frame = ss_all[fi] if ss_all is not None else None
                frames_rgb.append(
                    render_frame(coords[fi], labels[render_i], axis_limits,
                                 ss_labels=ss_frame, dpi=dpi, fig_size=fig_size)
                )
                if (render_i + 1) % 20 == 0:
                    logger.debug("  matplotlib: %d / %d frames", render_i + 1, len(kept))
        save_gif(frames_rgb, output_dir / gif_name, fps)

    # ── Mode 1: single GIF ─────────────────────────────────────────────────
    if n_states is None or n_states <= 1:
        keep_mask = np.ones(n_frames_total, dtype=bool)
        dists = np.zeros(n_frames_total, dtype=float)
        n_filtered = 0

        if rmsd_cutoff is not None:
            dists = frame_rmsd_to_ref(coords, ref_idx=0)
            keep_mask = dists <= rmsd_cutoff
            n_filtered = int(np.sum(~keep_mask))
            if n_filtered:
                logger.info(
                    "Filtered %d/%d frames (Cα RMSD > %.2f Å to frame 0)",
                    n_filtered, n_frames_total, rmsd_cutoff,
                )

        kept_indices = [i for i in range(n_frames_total) if keep_mask[i]]
        n_kept = len(kept_indices)
        labels = [
            f"{stem}  |  frame {frame_indices[fi]}"
            + (f"  RMSD={dists[fi]:.2f} Å" if rmsd_cutoff else "")
            + f"\n{ri + 1}/{n_kept}  •  L={n_ca} res  •  {renderer}"
            for ri, fi in enumerate(kept_indices)
        ]

        logger.info("Rendering %d frames with %s…", n_kept, renderer)
        _render_and_save(kept_indices, labels, f"{stem}.gif")

        if rmsd_cutoff is not None:
            write_report(output_dir, f"{stem}_frames.csv",
                         frame_indices, np.zeros(n_frames_total, int),
                         dists, keep_mask, rmsd_cutoff)

    # ── Mode 2: per-state GIFs ─────────────────────────────────────────────
    else:
        cutoff = rmsd_cutoff if rmsd_cutoff is not None else float("inf")

        logger.info("Computing dRMSD fingerprints for %d frames…", n_frames_total)
        features = np.vstack([coords_to_fingerprint(coords[i]) for i in range(n_frames_total)])

        proto_indices = select_prototypes(features, n_states)
        actual_states = len(proto_indices)
        logger.info("Prototypes (frame indices): %s", [frame_indices[i] for i in proto_indices])

        assignments, min_dists = assign_to_states(features, features[proto_indices])
        keep_mask = min_dists <= cutoff
        n_filtered = int(np.sum(~keep_mask))
        if n_filtered:
            logger.info("Filtered %d/%d frames (dRMSD > %.2f Å)", n_filtered, n_frames_total, cutoff)

        write_report(output_dir, f"{stem}_states.csv",
                     frame_indices, assignments, min_dists, keep_mask, cutoff)

        for state_idx in range(actual_states):
            state_mask = (assignments == state_idx) & keep_mask
            si = [i for i in range(n_frames_total) if state_mask[i]]

            if len(si) < min_frames:
                logger.warning("State %d: %d frame(s) after filtering (min=%d) — skipping",
                               state_idx + 1, len(si), min_frames)
                continue

            si_arr = np.array(si)
            order = np.argsort(min_dists[si_arr])
            si_sorted = si_arr[order].tolist()
            state_dists = min_dists[np.array(si_sorted)]
            n_state = len(si_sorted)

            logger.info("State %d: %d frames  dRMSD %.2f – %.2f Å",
                        state_idx + 1, n_state,
                        float(state_dists[0]), float(state_dists[-1]))

            labels = [
                f"{stem}  |  State {state_idx + 1}/{actual_states}  •  {renderer}\n"
                f"frame {frame_indices[fi]}  •  dRMSD={state_dists[ri]:.2f} Å  •  {ri + 1}/{n_state}"
                for ri, fi in enumerate(si_sorted)
            ]
            _render_and_save(si_sorted, labels, f"{stem}_state_{state_idx + 1:02d}.gif")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 56)
    print("  traj_gif_maker — Complete")
    print("=" * 56)
    print(f"  Protein          : {stem}  ({n_ca} residues)")
    print(f"  Renderer         : {renderer}")
    print(f"  Total frames     : {n_frames_total}  (stride={stride})")
    if n_states and n_states > 1:
        print(f"  States           : {actual_states}")
        if rmsd_cutoff:
            print(f"  Filtered frames  : {n_filtered}  (dRMSD > {cutoff:.2f} Å)")
    elif rmsd_cutoff:
        print(f"  Filtered frames  : {n_filtered}  (RMSD > {rmsd_cutoff:.2f} Å)")
    print(f"  Output dir       : {output_dir.resolve()}")
    print("=" * 56)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an animated GIF from a PDB topology + trajectory file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument("--pdb",  required=True, type=Path, metavar="FILE",
                        help="PDB topology file")
    parser.add_argument("--traj", required=True, type=Path, metavar="FILE",
                        help="Trajectory file (XTC, DCD, TRR, …)")

    # Output
    parser.add_argument("--output_dir", type=Path, default=Path("./traj_gifs"),
                        metavar="DIR", help="Directory for output GIFs and CSVs")

    # Renderer
    parser.add_argument(
        "--renderer", default="pymol", choices=["pymol", "matplotlib"],
        help=(
            "'pymol' (default) — cartoon with helix/sheet/loop coloring via PyMOL dss. "
            "'matplotlib' — simple Cα backbone trace coloured N→C."
        ),
    )
    parser.add_argument("--width",  type=int, default=600,
                        help="Frame width in pixels  (PyMOL renderer)")
    parser.add_argument("--height", type=int, default=600,
                        help="Frame height in pixels  (PyMOL renderer)")

    # Sampling
    parser.add_argument("--stride", type=int, default=1, metavar="N",
                        help="Use every N-th frame (1 = all frames)")
    parser.add_argument("--no_align", action="store_true",
                        help="Skip Cα superposition of frames onto frame 0")

    # Filtering / clustering
    parser.add_argument("--n_states", type=int, default=None, metavar="N",
                        help="Cluster into N states, write one GIF per state")
    parser.add_argument("--rmsd_cutoff", type=float, default=None, metavar="Å",
                        help=(
                            "Filter frames beyond this Å from reference. "
                            "Single-GIF mode: Cα RMSD to frame 0. "
                            "State mode: dRMSD to state prototype."
                        ))

    # GIF
    parser.add_argument("--fps",        type=float, default=5.0,  help="GIF frames per second")
    parser.add_argument("--min_frames", type=int,   default=2,    help="Min frames for a state GIF")
    parser.add_argument("--dpi",        type=int,   default=100,  help="DPI  (matplotlib renderer)")
    parser.add_argument("--fig_size",   type=float, default=6.0,  metavar="INCHES",
                        help="Figure size in inches  (matplotlib renderer)")

    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s  %(message)s",
    )

    run(
        pdb_path=args.pdb,
        traj_path=args.traj,
        output_dir=args.output_dir,
        stride=args.stride,
        align_frames=not args.no_align,
        n_states=args.n_states,
        rmsd_cutoff=args.rmsd_cutoff,
        fps=args.fps,
        min_frames=args.min_frames,
        renderer=args.renderer,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        fig_size=args.fig_size,
    )


if __name__ == "__main__":
    main()
