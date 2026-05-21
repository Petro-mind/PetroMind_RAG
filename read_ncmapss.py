#   1. Memory: reads unit by unit using h5py slicing
#   2. Health params: T array stores DELTAS not absolute values
#      delta < 0 means degraded (efficiency reduced)
#      delta = 0 means healthy (no change from baseline)

import os
import h5py
import numpy as np

from config import (
    SCENARIO_COLS, SENSOR_COLS, HEALTH_COLS,
    UNIT_FAILURE_MODE, ALERT_ZONE_CYCLES, CYCLE_SAMPLE_EVERY,
)

SENSOR_UNITS = {
    "Wf": "pps", "Nf": "rpm", "Nc": "rpm",
    "T24": "R",  "T30": "R",  "T48": "R",  "T50": "R",
    "P15": "psia", "P21": "psia", "P24": "psia",
    "Ps30": "psia", "P40": "psia", "P50": "psia",
}
SCENARIO_UNITS = {"alt": "ft", "Mach": "", "TRA": "%", "T2": "R"}


def _zone_label(rul: float) -> str:
    if rul > 200:
        return "healthy"
    elif rul > ALERT_ZONE_CYCLES:
        return "degrading"
    else:
        return "critical (alert zone)"


def _fmt_physical(means, stds, maxs) -> str:
    lines = []
    n = min(len(SENSOR_COLS), len(means))
    for i in range(n):
        col = SENSOR_COLS[i]
        u   = SENSOR_UNITS.get(col, "")
        lines.append(
            f"  {col:<14} mean={means[i]:.4f}{u}  "
            f"std={stds[i]:.4f}  max={maxs[i]:.4f}"
        )
    return "\n".join(lines)


def _fmt_virtual(means, stds, maxs) -> str:
    lines = []
    for i in range(len(means)):
        lines.append(
            f"  xv_{i:<12} mean={means[i]:.4f}  "
            f"std={stds[i]:.4f}  max={maxs[i]:.4f}"
        )
    return "\n".join(lines)


def _fmt_health(means, stds) -> str:
    """
    T array stores DELTA modifiers (deviation from baseline).
    delta < 0  → component efficiency REDUCED  → DEGRADED
    delta = 0  → no change from baseline       → healthy
    delta > 0  → efficiency improved (rare)
    """
    lines = []
    n = min(len(HEALTH_COLS), len(means))
    for i in range(n):
        col   = HEALTH_COLS[i]
        delta = means[i]
        # Degraded if efficiency modifier is negative (reduced from baseline)
        if delta < -0.0001:
            flag = f" [DEGRADED — delta={delta:.6f}]"
        elif delta == 0.0:
            flag = " [healthy]"
        else:
            flag = f" [delta={delta:.6f}]"
        lines.append(
            f"  {col:<20} mean_delta={delta:.6f}  std={stds[i]:.6f}{flag}"
        )
    return "\n".join(lines)


def _cycle_to_text(
    unit_id, cycle_id, rul, failure_mode,
    w_means,
    xs_means, xs_stds, xs_maxs,
    xv_means, xv_stds, xv_maxs,
    t_means,  t_stds,  t_maxs,
    n_rows,
) -> str:
    zone = _zone_label(rul)
    scen_parts = [
        f"{col}={w_means[i]:.1f}{SCENARIO_UNITS.get(col,'')}"
        for i, col in enumerate(SCENARIO_COLS)
    ]
    text = (
        f"SENSOR LOG ENTRY\n"
        f"================\n"
        f"unit_id      : {unit_id}\n"
        f"flight_cycle : {cycle_id}\n"
        f"rul          : {rul:.1f} cycles remaining\n"
        f"zone         : {zone}\n"
        f"failure_mode : {failure_mode.replace('_', ' ')}\n"
        f"n_samples    : {n_rows} (1 Hz readings)\n"
        f"\n"
        f"SCENARIO\n"
        f"--------\n"
        f"{'  '.join(scen_parts)}\n"
        f"\n"
        f"PHYSICAL SENSORS X_s ({len(xs_means)} vars)\n"
        f"------------------------------------------\n"
        f"{_fmt_physical(xs_means, xs_stds, xs_maxs)}\n"
        f"\n"
        f"VIRTUAL SENSORS X_v ({len(xv_means)} vars)\n"
        f"------------------------------------------\n"
        f"{_fmt_virtual(xv_means, xv_stds, xv_maxs)}\n"
        f"\n"
        f"HEALTH PARAMETERS T ({min(len(HEALTH_COLS), len(t_means))} vars)\n"
        f"(delta < 0 means component efficiency degraded vs baseline)\n"
        f"--------------------------------------------------------------\n"
        f"{_fmt_health(t_means, t_stds)}"
    )
    return text


def extract_engine_events(
    h5_path: str,
    split: str = "dev",
    sample_every: int = None,
) -> list:
    """
    Read N-CMAPSS HDF5 file one unit at a time — low memory usage.
    Reads only index array A first, then loads rows per cycle via slicing.
    """
    every  = sample_every if sample_every is not None else CYCLE_SAMPLE_EVERY
    fname  = os.path.basename(h5_path)
    suffix = f"_{split}"

    print(f"\n[read_ncmapss] {fname}  split='{split}'  sample_every={every}")

    # Load only index arrays first (small)
    with h5py.File(h5_path, "r") as h5:
        A_key = f"A{suffix}"
        Y_key = f"Y{suffix}"

        if A_key not in h5 or Y_key not in h5:
            print(f"  ERROR: split '{split}' not found.")
            return []

        A = h5[A_key][:]
        Y = h5[Y_key][:].flatten()

        # Print shapes for info
        for key in [f"W{suffix}", f"X_s{suffix}", f"X_v{suffix}", f"T{suffix}"]:
            if key in h5:
                shape = h5[key].shape
                mem   = shape[0] * shape[1] * 8 / 1e6
                print(f"  {key:<16} shape={shape}  mem~{mem:.0f}MB")

    unit_ids = np.unique(A[:, 0]).astype(int)
    print(f"  Unit IDs in file: {unit_ids.tolist()}")

    records = []

    for uid in unit_ids:
        unit_mask   = A[:, 0] == uid
        row_indices = np.where(unit_mask)[0]
        cycles      = np.unique(A[unit_mask, 1]).astype(int)

        sampled = list(cycles[::every])
        if len(cycles) > 0 and cycles[-1] not in sampled:
            sampled.append(int(cycles[-1]))

        failure_mode = UNIT_FAILURE_MODE.get(int(uid), "unknown")
        kept = 0

        for cyc in sampled:
            cyc_mask_local  = A[unit_mask, 1] == cyc
            cyc_row_indices = row_indices[cyc_mask_local]

            if len(cyc_row_indices) == 0:
                continue

            n_rows = len(cyc_row_indices)
            rul    = float(Y[cyc_row_indices[-1]])

            # Load only rows for this cycle
            with h5py.File(h5_path, "r") as h5:
                w_rows  = h5[f"W{suffix}"][cyc_row_indices]
                xs_rows = h5[f"X_s{suffix}"][cyc_row_indices]

                xv_key  = f"X_v{suffix}"
                xv_rows = h5[xv_key][cyc_row_indices] if xv_key in h5 else None

                t_key   = f"T{suffix}"
                t_rows  = h5[t_key][cyc_row_indices] if t_key in h5 else None

            w_means  = w_rows.mean(axis=0)
            xs_means = xs_rows.mean(axis=0)
            xs_stds  = xs_rows.std(axis=0)
            xs_maxs  = xs_rows.max(axis=0)

            if xv_rows is not None:
                xv_means = xv_rows.mean(axis=0)
                xv_stds  = xv_rows.std(axis=0)
                xv_maxs  = xv_rows.max(axis=0)
            else:
                xv_means = xv_stds = xv_maxs = np.zeros(0)

            if t_rows is not None:
                t_means = t_rows.mean(axis=0)
                t_stds  = t_rows.std(axis=0)
                t_maxs  = t_rows.max(axis=0)
            else:
                t_means = t_stds = t_maxs = np.zeros(0)

            text = _cycle_to_text(
                unit_id=int(uid),      cycle_id=int(cyc),
                rul=rul,               failure_mode=failure_mode,
                w_means=w_means,
                xs_means=xs_means,     xs_stds=xs_stds,    xs_maxs=xs_maxs,
                xv_means=xv_means,     xv_stds=xv_stds,    xv_maxs=xv_maxs,
                t_means=t_means,       t_stds=t_stds,       t_maxs=t_maxs,
                n_rows=n_rows,
            )

            records.append({
                "unit_id":      int(uid),
                "cycle_id":     int(cyc),
                "rul":          rul,
                "failure_mode": failure_mode,
                "zone":         _zone_label(rul),
                "text":         text,
            })
            kept += 1

        print(f"  unit {uid:>3}  total_cycles={len(cycles):>4}  "
              f"kept={kept:>3}  mode={failure_mode}")

    print(f"[read_ncmapss] Done — {len(records)} records from {fname}\n")
    return records