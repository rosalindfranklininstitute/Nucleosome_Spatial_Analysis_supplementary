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

import subprocess
import re
import os
import glob
import tempfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def extract_specimen_id(path):
    """Extracts specimen ID e.g. J273-1_TS_01 or J273-1_TS_01_2."""
    match = re.search(r'(J\d+-\d+_TS_\d+(?:_\d+)?)', path)
    return match.group(1) if match else None


def find_raw_denoised_pairs(raw_dir, denoised_dir):
    """
    Pairs .star files from two directories by specimen ID.
    Reports unmatched files and warns on duplicate IDs.
    """
    raw_pattern  = os.path.join(raw_dir,  '*', '*.star')
    denoised_pattern = os.path.join(denoised_dir, '*', '*.star')

    def build_dict(pattern):
        d = {}
        for f in glob.glob(pattern):
            sid = extract_specimen_id(f)
            if not sid:
                log.warning(f"Could not extract specimen ID from: {f}")
                continue
            if sid in d:
                log.warning(f"Duplicate specimen ID '{sid}': {d[sid]} vs {f} - keeping first")
            else:
                d[sid] = f
        return d

    raw_dict  = build_dict(raw_pattern)
    denoised_dict = build_dict(denoised_pattern)
    log.info(f"Found {len(raw_dict)} raw files, {len(denoised_dict)} denoised files")

    paired = []
    unmatched_denoised = []

    for sid, denoised_file in denoised_dict.items():
        if sid in raw_dict:
            paired.append((denoised_file, raw_dict[sid]))
        else:
            base_id = re.sub(r'_\d+$', '', sid)
            if base_id in raw_dict:
                log.info(f"Fuzzy match: '{sid}' -> '{base_id}'")
                paired.append((denoised_file, raw_dict[base_id]))
            else:
                unmatched_denoised.append(denoised_file)

    matched_raw_ids = {extract_specimen_id(p[1]) for p in paired}
    unmatched_raw   = [f for sid, f in raw_dict.items() if sid not in matched_raw_ids]

    if unmatched_denoised:
        log.warning(f"{len(unmatched_denoised)} denoised files with no raw match:")
        for f in unmatched_denoised:
            log.warning(f"  {f}")
    if unmatched_raw:
        log.warning(f"{len(unmatched_raw)} raw files with no denoised match:")
        for f in unmatched_raw:
            log.warning(f"  {f}")

    return paired


def trim_star_file_columns(input_file, num_columns=15):
    """
    Trims STAR file to first num_columns data columns, but always preserves
    coordinate and RELION angle columns required by the ellipsoidal overlap script.

    Required preserved columns:
      - CoordinateX, CoordinateY, CoordinateZ
      - AngleRot, AngleTilt, AnglePsi
    """
    required_patterns = [
        "CoordinateX", "CoordinateY", "CoordinateZ",
        "AngleRot", "AngleTilt", "AnglePsi",
    ]

    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # First pass: determine which column indices to keep
    keep_indices = []
    header_lines = []
    col_counter = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('data_') or stripped.startswith('loop_'):
            col_counter = 0
            continue
        if stripped.startswith('_rln'):
            col_counter += 1
            keep_this = col_counter <= num_columns or any(p in stripped for p in required_patterns)
            if keep_this:
                keep_indices.append(col_counter - 1)
                header_lines.append(stripped)

    if not keep_indices:
        raise ValueError(f"No STAR columns selected from {input_file}")

    header_text = "\n".join(header_lines)
    missing = [p for p in required_patterns if p not in header_text]
    if missing:
        raise ValueError(
            f"Missing required columns in {input_file}: {missing}. "
            f"The ellipsoidal overlap script requires coordinates and RELION angles."
        )

    # Second pass: write trimmed file
    trimmed_file = tempfile.NamedTemporaryFile(delete=False, suffix='.star', mode='w')
    col_counter = 0
    output_col_counter = 0

    with open(input_file, 'r') as infile:
        for line in infile:
            stripped = line.strip()
            if not stripped:
                trimmed_file.write(line)
            elif stripped.startswith('data_') or stripped.startswith('loop_'):
                col_counter = 0
                output_col_counter = 0
                trimmed_file.write(line)
            elif stripped.startswith('_rln'):
                col_counter += 1
                if (col_counter - 1) in keep_indices:
                    output_col_counter += 1
                    renumbered = re.sub(r'#\d+', f'#{output_col_counter}', stripped)
                    trimmed_file.write(renumbered + '\n')
            elif stripped.startswith('#'):
                trimmed_file.write(line)
            else:
                cols = stripped.split()
                trimmed_cols = [cols[i] for i in keep_indices if i < len(cols)]
                trimmed_file.write(' '.join(trimmed_cols) + '\n')

    trimmed_file.close()
    return trimmed_file.name


# ==== Configuration ====
raw_dir          = "demo_unionise_raw_dir"
denoised_dir      = "demo_unionise_denoised_dir"
unionise_script   = "src/unionise_star.py"
output_dir        = "demo_unionise_outputs"
priority_source   = "raw"   # "raw" or "denoised"

# Ellipsoidal overlap parameters
axial_threshold   = 12.0   # px
radial_threshold  = 24.0   # px
centre_gate       = 27.0   # px hard nearest-neighbour gate
overlap_threshold = 0.1    # fraction
overlap_mode      = "symmetric_max"
n_points          = 2000
n_jobs            = -1
debug_mode        = True
# ========================


file_pairs = find_raw_denoised_pairs(raw_dir, denoised_dir)
log.info(f"Found {len(file_pairs)} matched STAR file pairs.")

success_count, error_count = 0, 0

for denoised_file, raw_file in file_pairs:
    raw_id  = extract_specimen_id(raw_file)
    denoised_id = extract_specimen_id(denoised_file)

    if raw_id != denoised_id:
        log.error(f"ID mismatch after pairing: {raw_id} vs {denoised_id} - skipping")
        error_count += 1
        continue

    specimen_id = raw_id
    log.info(f"Processing {specimen_id}:\n  raw : {raw_file}\n  denoised: {denoised_file}")

    trimmed_raw = None
    trimmed_denoised = None

    try:
        trimmed_raw  = trim_star_file_columns(raw_file, 15)
        trimmed_denoised = trim_star_file_columns(denoised_file, 15)

        if priority_source == "raw":
            file_priority = trimmed_raw
            file_other = trimmed_denoised
        else:
            file_priority = trimmed_denoised
            file_other = trimmed_raw

        output_file = os.path.join(
            output_dir,
            (
                f"{specimen_id}_union_"
                f"ellipsoid_ax{axial_threshold}_rad{radial_threshold}_"
                f"cg{centre_gate}_ov{overlap_threshold}.star"
            )
        )

        command = [
            unionise_script,
            "--priority",          file_priority,
            "--other",             file_other,
            "--output",            output_file,
            "--axial-threshold",   str(axial_threshold),
            "--radial-threshold",   str(radial_threshold),
            "--centre-gate",       str(centre_gate),
            "--overlap-threshold", str(overlap_threshold),
            "--overlap-mode",      overlap_mode,
            "--n-points",          str(n_points),
            "--n-jobs",            str(n_jobs),
        ]

        if not debug_mode:
            command.append("--no-debug")

        log.info(f"Command: {' '.join(command)}")

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        if result.stdout:
            log.info(result.stdout)
        if result.stderr:
            log.warning(result.stderr)

        log.info(f"Success: {output_file}")
        success_count += 1

    except subprocess.CalledProcessError as e:
        log.error(f"Failed for {specimen_id}: {e}\nstdout: {e.stdout}\nstderr: {e.stderr}")
        error_count += 1

    except Exception as e:
        log.error(f"Unexpected error for {specimen_id}: {e}")
        error_count += 1

    finally:
        if trimmed_raw and os.path.exists(trimmed_raw):
            os.remove(trimmed_raw)
        if trimmed_denoised and os.path.exists(trimmed_denoised):
            os.remove(trimmed_denoised)

log.info(f"Done. {success_count} succeeded, {error_count} failed.")
