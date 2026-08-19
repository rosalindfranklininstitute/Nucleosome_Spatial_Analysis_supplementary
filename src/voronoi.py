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

# Co-Authored-By: ChatGPT GPT-5 noreply@openai.com

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Voronoi
import starfile


# -----------------------------
# Style
# -----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
})


# -----------------------------
# TAB20C colour order
# -----------------------------
TAB20C_ORDER = [
    6, 10, 2, 14, 18, 5, 9, 1, 13, 17,
    8, 12, 4, 16, 19, 7, 11, 3, 15, 18,
]


def get_tab20c_colors(n):
    base = plt.cm.tab20c.colors
    return [base[TAB20C_ORDER[i % len(TAB20C_ORDER)]] for i in range(n)]


# -----------------------------
# Load STAR coordinates
# -----------------------------
def load_coords(path):
    data = starfile.read(path)
    df = list(data.values())[0] if isinstance(data, dict) else data

    def get_col(name):
        if name in df.columns:
            return df[name].values

        alt = name.lstrip("_")
        if alt in df.columns:
            return df[alt].values

        raise KeyError(f"Missing coordinate column: {name}")

    return np.column_stack([
        get_col("_rlnCoordinateX"),
        get_col("_rlnCoordinateY"),
        get_col("_rlnCoordinateZ"),
    ])


# -----------------------------
# 3D Voronoi-based local density
# -----------------------------
def voronoi_density(xyz_nm):
    tessellation = Voronoi(xyz_nm)
    volumes = np.full(len(xyz_nm), np.nan, dtype=float)

    for i in range(len(xyz_nm)):
        region = tessellation.regions[tessellation.point_region[i]]

        # Regions containing -1 are unbounded.
        if not region or -1 in region:
            continue

        try:
            volumes[i] = ConvexHull(tessellation.vertices[region]).volume
        except Exception:
            continue

    density = np.full_like(volumes, np.nan)
    valid = np.isfinite(volumes) & (volumes > 0)
    density[valid] = 1.0 / volumes[valid]
    return density


# -----------------------------
# Empirical CDF on a shared grid
# -----------------------------
def empirical_cdf_on_grid(values, x_grid):
    values = np.sort(values)
    return np.searchsorted(values, x_grid, side="right") / len(values)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--group", action="append", nargs="+", required=True)
    parser.add_argument("--angpix", type=float, required=True)

    parser.add_argument("--full-x", type=float, required=True)
    parser.add_argument("--full-y", type=float, required=True)
    parser.add_argument("--full-z", type=float, required=True)

    parser.add_argument("--crop-x", type=float, required=True)
    parser.add_argument("--crop-y", type=float, required=True)
    parser.add_argument("--crop-z", type=float, required=True)

    parser.add_argument("--grid-points", type=int, default=200)
    parser.add_argument("--out-prefix", default="voronoi")
    parser.add_argument("--out-dir", default=".")

    args = parser.parse_args()

    if args.grid_points < 2:
        raise ValueError("--grid-points must be at least 2.")

    pixel_size_nm = args.angpix / 10.0
    os.makedirs(args.out_dir, exist_ok=True)

    def center_bounds(full, crop):
        start = (full - crop) / 2.0
        return start, start + crop

    x0, x1 = center_bounds(args.full_x, args.crop_x)
    y0, y1 = center_bounds(args.full_y, args.crop_y)
    z0, z1 = center_bounds(args.full_z, args.crop_z)

    # Store one density array per valid tomogram.
    group_data = []

    for group in args.group:
        group_name = group[0]
        paths = []

        for pattern in group[1:]:
            paths.extend(glob.glob(pattern))

        tomogram_densities = []

        for path in paths:
            xyz = load_coords(path)

            crop_mask = (
                (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1) &
                (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
            )
            xyz = xyz[crop_mask]

            if len(xyz) < 20:
                print(f"Skipping {path}: fewer than 20 particles after cropping")
                continue

            density_nm3 = voronoi_density(xyz * pixel_size_nm)
            density_nm3 = density_nm3[np.isfinite(density_nm3)]

            if len(density_nm3) == 0:
                print(f"Skipping {path}: no valid bounded Voronoi cells")
                continue

            tomogram_densities.append(density_nm3)

        if tomogram_densities:
            group_data.append((group_name, tomogram_densities))
        else:
            print(f"Skipping group {group_name}: no valid tomograms")

    if not group_data:
        raise ValueError("No valid tomograms were available for CDF analysis.")

    global_max_density = max(
        density.max()
        for _, tomograms in group_data
        for density in tomograms
    )
    x_common = np.linspace(0.0, global_max_density, args.grid_points)

    colors = get_tab20c_colors(len(group_data))
    fig, ax = plt.subplots(figsize=(6, 5))

    for group_index, (group_name, tomograms) in enumerate(group_data):
        cdfs = np.vstack([
            empirical_cdf_on_grid(density, x_common)
            for density in tomograms
        ])

        mean_cdf = np.mean(cdfs, axis=0)

        if len(tomograms) > 1:
            sem_cdf = np.std(cdfs, axis=0, ddof=1) / np.sqrt(len(tomograms))
        else:
            sem_cdf = np.zeros_like(mean_cdf)

        ax.plot(
            x_common,
            mean_cdf,
            color=colors[group_index],
            label=group_name,
        )
        ax.fill_between(
            x_common,
            mean_cdf - sem_cdf,
            mean_cdf + sem_cdf,
            color=colors[group_index],
            alpha=0.3,
        )

    ax.set_xlabel(r"Voronoi-based local density ($nm^{-3}$)")
    ax.set_ylabel("Cumulative distribution function")
    ax.set_title("Mean CDF ± SEM")
    ax.legend()

    fig.tight_layout()
    output_path = os.path.join(
        args.out_dir,
        f"{args.out_prefix}_CDF_nm3.svg",
    )
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
