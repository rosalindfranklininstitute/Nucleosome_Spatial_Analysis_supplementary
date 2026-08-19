#!/usr/bin/env python3

"""
   Copyright [2026] [Rosalind Franklin Institute]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Co-Authored-By: Claude Sonnet 5 noreply@anthropic.com

import argparse
import numpy as np
import pandas as pd
import starfile
from scipy.spatial import cKDTree
from joblib import Parallel, delayed


def find_coord_columns(df):
    coord_x = next((col for col in df.columns if 'CoordinateX' in col), None)
    coord_y = next((col for col in df.columns if 'CoordinateY' in col), None)
    coord_z = next((col for col in df.columns if 'CoordinateZ' in col), None)
    if not (coord_x and coord_y and coord_z):
        raise ValueError(f"Could not find coordinate columns in {df.columns.tolist()}")
    return [coord_x, coord_y, coord_z]


def find_angle_columns(df):
    angle_rot = next((col for col in df.columns if 'AngleRot' in col), None)
    angle_tilt = next((col for col in df.columns if 'AngleTilt' in col), None)
    angle_psi = next((col for col in df.columns if 'AnglePsi' in col), None)
    if not (angle_rot and angle_tilt and angle_psi):
        raise ValueError(f"Could not find angle columns in {df.columns.tolist()}")
    return [angle_rot, angle_tilt, angle_psi]


def relion_cylinder_axes(df):
    """
    Convert RELION Rot/Tilt into unit cylinder axis vectors.
    Template flat face normal is along Z at Tilt=0, Rot=0.
    Psi is ignored because of circular symmetry around axis.
    """
    angle_cols = find_angle_columns(df)
    angles = df[angle_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
    n_bad = np.isnan(angles).any(axis=1).sum()
    if n_bad > 0:
        raise ValueError(f"Found {n_bad} rows with non-numeric angles")
    rot = np.deg2rad(angles[:, 0])
    tilt = np.deg2rad(angles[:, 1])
    axes = np.column_stack([
        np.sin(tilt) * np.cos(rot),
        np.sin(tilt) * np.sin(rot),
        np.cos(tilt)
    ])
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    return axes / norms


def make_template_cylinder_points(radius, height, n_points=2000, seed=42):
    """
    Uniform random points filling a cylinder volume.
    Axis along Z, centre at origin, from -height/2 to +height/2.
    """
    rng = np.random.default_rng(seed)
    n_sample = int(n_points / 0.785) + 200
    while True:
        x = rng.uniform(-radius, radius, n_sample)
        y = rng.uniform(-radius, radius, n_sample)
        z = rng.uniform(-height / 2.0, height / 2.0, n_sample)
        inside = (x**2 + y**2) <= radius**2
        pts = np.column_stack([x[inside], y[inside], z[inside]])
        if len(pts) >= n_points:
            return pts[:n_points].astype(np.float32)
        n_sample = int(n_sample * 1.5)


def precompute_oriented_cylinders(coords, axes, template_points):
    """
    Vectorised orientation of all cylinders.
    template X -> in-plane u
    template Y -> in-plane v
    template Z -> cylinder axis
    Returns (N, n_pts, 3)
    """
    n = len(axes)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    mask = np.abs(axes[:, 2]) >= 0.9
    ref[mask] = np.array([1.0, 0.0, 0.0])

    u = np.cross(ref, axes)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(axes, u)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    all_points = (
        coords[:, None, :]
        + template_points[None, :, 0:1] * u[:, None, :]
        + template_points[None, :, 1:2] * v[:, None, :]
        + template_points[None, :, 2:3] * axes[:, None, :]
    )
    return all_points.astype(np.float32)


def point_inside_ellipsoid_cylinder(points, coord, axis, axial_threshold, radial_threshold):
    """
    Ellipsoidal duplicate test with different thresholds for axial vs radial directions.

    A point is considered inside if:
        (radial_dist / radial_threshold)^2 + (axial_dist / axial_threshold)^2 < 1

    Returns a boolean array for the sampled points.
    """
    v = points - coord
    axial_dist = v @ axis
    axial_comp = axial_dist[:, None] * axis
    radial_dist = np.sqrt(np.sum((v - axial_comp) ** 2, axis=1))

    inside = (
        (radial_dist / radial_threshold) ** 2 +
        (axial_dist / axial_threshold) ** 2
    ) < 1.0

    return inside


def ellipsoid_overlap_fraction(oriented_a, coord_a, axis_a,
                               oriented_b, coord_b, axis_b,
                               axial_threshold, radial_threshold,
                               mode="symmetric_max"):
    """
    Overlap fraction using ellipsoidal duplicate test.
    b_in_a: fraction of B's points inside A's ellipsoid
    a_in_b: fraction of A's points inside B's ellipsoid
    """
    b_in_a = point_inside_ellipsoid_cylinder(
        oriented_b, coord_a, axis_a, axial_threshold, radial_threshold
    ).mean()

    if mode == "b_in_a":
        return b_in_a

    a_in_b = point_inside_ellipsoid_cylinder(
        oriented_a, coord_b, axis_b, axial_threshold, radial_threshold
    ).mean()

    if mode == "a_in_b":
        return a_in_b
    if mode == "symmetric_max":
        return max(a_in_b, b_in_a)
    if mode == "symmetric_min":
        return min(a_in_b, b_in_a)
    if mode == "symmetric_mean":
        return 0.5 * (a_in_b + b_in_a)

    raise ValueError(f"Unknown overlap mode: {mode}")


def dedupe_within_file(df, centre_gate, axial_threshold, radial_threshold, overlap_threshold,
                       template_points, overlap_mode="symmetric_max",
                       score_col=None, debug=False):
    """
    Remove duplicates within a single STAR file using ellipsoidal overlap.

    Hard gate:
      if a particle has no neighbour within centre_gate, it is kept.
    """
    coord_cols = find_coord_columns(df)
    coords = df[coord_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
    axes = relion_cylinder_axes(df)

    n_bad = np.isnan(coords).any(axis=1).sum()
    if n_bad > 0:
        raise ValueError(f"Found {n_bad} rows with non-numeric coordinates")

    order = (df[score_col].argsort()[::-1].to_numpy()
             if score_col and score_col in df.columns
             else np.arange(len(df)))

    coords_sorted = coords[order]
    axes_sorted = axes[order]
    oriented = precompute_oriented_cylinders(coords_sorted, axes_sorted, template_points)

    tree = cKDTree(coords_sorted)
    accepted_mask = np.ones(len(coords_sorted), dtype=bool)

    for i, point in enumerate(coords_sorted):
        if not accepted_mask[i]:
            continue

        neighbours = [
            nb for nb in tree.query_ball_point(point, r=centre_gate)
            if nb != i and accepted_mask[nb]
        ]

        if not neighbours:
            # Isolated particle: keep it.
            continue

        for nb in neighbours:
            frac = ellipsoid_overlap_fraction(
                oriented_a=oriented[i], coord_a=point, axis_a=axes_sorted[i],
                oriented_b=oriented[nb], coord_b=coords_sorted[nb], axis_b=axes_sorted[nb],
                axial_threshold=axial_threshold,
                radial_threshold=radial_threshold,
                mode=overlap_mode
            )
            if frac >= overlap_threshold:
                accepted_mask[nb] = False

    deduped_df = df.iloc[np.sort(order[accepted_mask])].copy()
    if debug:
        print(f"  [DEBUG] dedupe_within_file: {len(df)} -> {len(deduped_df)} "
              f"({len(df) - len(deduped_df)} removed)")
    return deduped_df


def _check_one_other_particle(i, candidates, coords_priority, axes_priority,
                               oriented_priority, coords_other, axes_other,
                               oriented_other, axial_threshold, radial_threshold,
                               overlap_threshold, overlap_mode):
    """Worker for parallel cross-file dedup."""
    best = 0.0
    remove = False
    for nb in candidates:
        frac = ellipsoid_overlap_fraction(
            oriented_a=oriented_priority[nb], coord_a=coords_priority[nb], axis_a=axes_priority[nb],
            oriented_b=oriented_other[i], coord_b=coords_other[i], axis_b=axes_other[i],
            axial_threshold=axial_threshold,
            radial_threshold=radial_threshold,
            mode=overlap_mode
        )
        if frac > best:
            best = frac
        if frac >= overlap_threshold:
            remove = True
    return remove, best


def unionise_star_files_live(
    file_priority, file_other, output,
    axial_threshold, radial_threshold, overlap_threshold,
    centre_gate=27.0,
    n_points=2000, overlap_mode="symmetric_max",
    score_col=None, debug=True, n_jobs=-1
):
    df_priority = starfile.read(file_priority)
    df_other = starfile.read(file_other)

    template_points = make_template_cylinder_points(
        radius=radial_threshold, height=axial_threshold * 2,
        n_points=n_points
    )

    print(f"\n[DIAG] Priority file      : {file_priority} ({len(df_priority)} particles)")
    print(f"[DIAG] Other file         : {file_other} ({len(df_other)} particles)")
    print(f"[DIAG] axial_threshold    : {axial_threshold} px")
    print(f"[DIAG] radial_threshold   : {radial_threshold} px")
    print(f"[DIAG] centre_gate        : {centre_gate} px")
    print(f"[DIAG] overlap_threshold  : {overlap_threshold}")
    print(f"[DIAG] overlap_mode       : {overlap_mode}")
    print(f"[DIAG] n_points/cylinder  : {n_points}")
    print(f"[DIAG] n_jobs             : {n_jobs}")

    print(f"\n[INFO] Deduplicating priority file...")
    df_priority = dedupe_within_file(
        df=df_priority,
        centre_gate=centre_gate,
        axial_threshold=axial_threshold,
        radial_threshold=radial_threshold,
        overlap_threshold=overlap_threshold,
        template_points=template_points,
        overlap_mode=overlap_mode,
        score_col=score_col, debug=debug
    )

    print(f"[INFO] Deduplicating other file...")
    df_other = dedupe_within_file(
        df=df_other,
        centre_gate=centre_gate,
        axial_threshold=axial_threshold,
        radial_threshold=radial_threshold,
        overlap_threshold=overlap_threshold,
        template_points=template_points,
        overlap_mode=overlap_mode,
        score_col=score_col, debug=debug
    )

    coord_cols = find_coord_columns(df_priority)
    coords_priority = df_priority[coord_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
    coords_other = df_other[coord_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
    axes_priority = relion_cylinder_axes(df_priority)
    axes_other = relion_cylinder_axes(df_other)

    oriented_priority = precompute_oriented_cylinders(coords_priority, axes_priority, template_points)
    oriented_other = precompute_oriented_cylinders(coords_other, axes_other, template_points)

    tree_priority = cKDTree(coords_priority)
    candidate_lists = tree_priority.query_ball_point(
        coords_other, r=centre_gate, workers=n_jobs
    )

    # Diagnostics
    n_no_candidates = sum(1 for c in candidate_lists if not c)
    print(f"\n[DIAG] candidate_gate                           : {centre_gate:.1f} px")
    print(f"[DIAG] Other particles with no priority candidates : {n_no_candidates} "
          f"({100*n_no_candidates/len(coords_other):.1f}%) -> kept without overlap test")
    print(f"[DIAG] Other particles tested for overlap          : {len(coords_other)-n_no_candidates} "
          f"({100*(len(coords_other)-n_no_candidates)/len(coords_other):.1f}%)")
    # END diagnostics

    print(f"[INFO] Cross-file deduplication "
          f"({len(coords_other)} other particles, n_jobs={n_jobs})...")

    has_candidates = [i for i, c in enumerate(candidate_lists) if c]

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_check_one_other_particle)(
            i=i,
            candidates=candidate_lists[i],
            coords_priority=coords_priority,
            axes_priority=axes_priority,
            oriented_priority=oriented_priority,
            coords_other=coords_other,
            axes_other=axes_other,
            oriented_other=oriented_other,
            axial_threshold=axial_threshold,
            radial_threshold=radial_threshold,
            overlap_threshold=overlap_threshold,
            overlap_mode=overlap_mode
        )
        for i in has_candidates
    )

    keep_mask = np.ones(len(coords_other), dtype=bool)
    best_overlap = np.zeros(len(coords_other), dtype=float)
    for idx, i in enumerate(has_candidates):
        remove, best = results[idx]
        keep_mask[i] = not remove
        best_overlap[i] = best

    df_other_kept = df_other.iloc[keep_mask].copy()
    combined_df = pd.concat([df_priority, df_other_kept], ignore_index=True)
    starfile.write(combined_df, output)

    print(f"\n[SUMMARY] {output}")
    print(f"  Priority kept      : {len(df_priority)}")
    print(f"  Other kept         : {keep_mask.sum()}")
    print(f"  Other removed      : {(~keep_mask).sum()}")
    print(f"  Total written      : {len(combined_df)}")
    print(f"  axial_threshold    : {axial_threshold} px")
    print(f"  radial_threshold   : {radial_threshold} px")
    print(f"  centre_gate        : {centre_gate} px")
    print(f"  overlap_threshold  : {overlap_threshold}")
    print(f"  overlap_mode       : {overlap_mode}")

    if debug:
        if (~keep_mask).sum() > 0:
            print(f"  [DEBUG] Min overlap (removed) : {best_overlap[~keep_mask].min():.3f}")
        if keep_mask.sum() > 0:
            print(f"  [DEBUG] Max overlap (kept)    : {best_overlap[keep_mask].max():.3f}")

        overlap_dist = np.array([best_overlap[i] for i in has_candidates])
        if len(overlap_dist) > 0:
            print(f"\n[DIAG] Overlap distribution (tested particles):")
            print(f"  min={overlap_dist.min():.3f}  max={overlap_dist.max():.3f}  "
                  f"mean={overlap_dist.mean():.3f}  median={np.median(overlap_dist):.3f}")
            for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                print(f"  >= {t:.1f} : {(overlap_dist >= t).mean()*100:.1f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unionise two RELION STAR files using orientation-aware ellipsoidal deduplication.\n"
            "Particles farther than centre_gate are kept without overlap testing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--priority", required=True, help="Priority STAR file.")
    parser.add_argument("--other", required=True, help="Other STAR file.")
    parser.add_argument("--output", required=True, help="Output STAR file.")

    parser.add_argument(
        "--axial-threshold",
        type=float,
        default=12.0,
        help="Half-height threshold in pixels (face-to-face direction). Default: 13.0"
    )
    parser.add_argument(
        "--radial-threshold",
        type=float,
        default=24.0,
        help="Radial threshold in pixels (side-by-side direction). Default: 14.0"
    )
    parser.add_argument(
        "--centre-gate",
        type=float,
        default=27.0,
        help="Nearest-neighbour centre-distance gate in pixels. Default: 27.0"
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.1,
        help="Fractional overlap to call duplicate. Default: 0.3"
    )
    parser.add_argument(
        "--overlap-mode",
        choices=["b_in_a", "a_in_b", "symmetric_max", "symmetric_min", "symmetric_mean"],
        default="symmetric_max",
        help="How to combine directional overlaps. Default: symmetric_max"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=2000,
        help="Volume sample points per particle. Default: 2000"
    )
    parser.add_argument(
        "--score-col",
        default=None,
        help="Score column for tiebreaking within each file."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs. -1 = all cores."
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Suppress debug output"
    )

    args = parser.parse_args()

    if not (0.0 < args.overlap_threshold <= 1.0):
        parser.error("--overlap-threshold must be between 0 and 1")
    if args.axial_threshold <= 0:
        parser.error("--axial-threshold must be positive")
    if args.radial_threshold <= 0:
        parser.error("--radial-threshold must be positive")
    if args.centre_gate <= 0:
        parser.error("--centre-gate must be positive")

    return args


def main():
    args = parse_args()
    unionise_star_files_live(
        file_priority=args.priority,
        file_other=args.other,
        output=args.output,
        axial_threshold=args.axial_threshold,
        radial_threshold=args.radial_threshold,
        overlap_threshold=args.overlap_threshold,
        centre_gate=args.centre_gate,
        n_points=args.n_points,
        overlap_mode=args.overlap_mode,
        score_col=args.score_col,
        debug=not args.no_debug,
        n_jobs=args.n_jobs
    )


if __name__ == "__main__":
    main()
