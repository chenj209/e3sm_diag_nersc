#!/usr/bin/env python3
"""
diurnal_similarity.py

Compute a single scalar similarity between two E3SM-Diags diurnal-cycle NetCDFs
using the first-harmonic complex vector correlation:

    rho = sum(w * F_A * conj(F_B)) / sqrt( sum(w*|F_A|^2) * sum(w*|F_B|^2) )
    with F = amplitude * exp(i * 2π * phase_hours / 24)

Outputs:
  - |rho| in [0,1]  (higher = more similar timing & strength)
  - mean_lag_h  (hours; B relative to A; negative => B earlier)

Usage:
  python diurnal_similarity.py caseA.nc caseB.nc \
      --var PRECT --amp_thresh 0.0 --weights area

Notes:
  * Files should be on the same grid (as produced by e3sm_diags after regridding).
  * Phase in HOURS [0,24), amplitude in mm/day.
"""

import argparse
import json
import math
import numpy as np
from netCDF4 import Dataset

def read_fields(path, var="PRECT"):
    with Dataset(path, "r") as ds:
        # coords
        lat = ds.variables["lat"][:].astype(float)  # (nlat,)
        # data
        amp = ds.variables[f"{var}_diurnal_amplitude"][:].astype(float)  # (nlat,nlon)
        ph  = ds.variables[f"{var}_diurnal_phase"][:].astype(float)      # (nlat,nlon), hours
    return lat, amp, ph

def area_weights(lat):
    # simple cos(lat) area weights, broadcast to (nlat, nlon)
    wlat = np.cos(np.deg2rad(lat))
    wlat[wlat < 0] = 0  # guard antipodal weirdness
    return wlat[:, None]

def build_complex(amp, phase_h):
    """F = A * exp(i*theta); theta = 2π * phase_hours / 24"""
    theta = 2 * np.pi * (phase_h / 24.0)
    return amp * (np.cos(theta) + 1j * np.sin(theta))

def circular_diff_hours(pB_h, pA_h):
    """Δphase wrapped to [-12, 12] hours."""
    return ((pB_h - pA_h + 12.0) % 24.0) - 12.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caseA", help="NetCDF from e3sm_diags for case A")
    ap.add_argument("caseB", help="NetCDF from e3sm_diags for case B")
    ap.add_argument("--var", default="PRECT", help="Base var name (default PRECT)")
    ap.add_argument("--amp_thresh", type=float, default=0.0,
                    help="Mask cells where either amplitude < thresh (mm/day)")
    ap.add_argument("--weights", choices=["area", "amp", "none"], default="area",
                    help="Weighting scheme for regional metric")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args()

    latA, ampA, phA = read_fields(args.caseA, args.var)
    latB, ampB, phB = read_fields(args.caseB, args.var)

    # Basic sanity
    if ampA.shape != ampB.shape or phA.shape != phB.shape:
        raise ValueError("Amplitude/phase grids differ between files.")

    # Build mask: finite and above amplitude threshold
    mask = np.isfinite(ampA) & np.isfinite(ampB) & np.isfinite(phA) & np.isfinite(phB)
    if args.amp_thresh > 0:
        mask &= (ampA >= args.amp_thresh) & (ampB >= args.amp_thresh)

    if not np.any(mask):
        raise ValueError("All cells masked out. Lower --amp_thresh or check data.")

    # Choose weights
    if args.weights == "area":
        w = area_weights(latA)
    elif args.weights == "amp":
        # precip-weighted by geometric mean amplitude (avoids favoring only one case)
        w = np.sqrt(ampA * ampB)
    else:
        w = np.ones_like(ampA)

    # Apply mask
    w = np.where(mask, w, 0.0)

    # Complex first-harmonic fields
    FA = build_complex(ampA, phA)
    FB = build_complex(ampB, phB)

    # Regional complex correlation
    num = np.nansum(w * FA * np.conj(FB))
    den = math.sqrt(np.nansum(w * np.abs(FA)**2) * np.nansum(w * np.abs(FB)**2))
    rho = num / den if den > 0 else np.nan + 1j*np.nan

    similarity = float(np.abs(rho))                 # [0,1]
    mean_lag_h = float((24.0/(2*np.pi)) * np.angle(rho))  # B relative to A, hours

    # Helpful secondary stats (don’t affect the main scalar)
    dphi_h = circular_diff_hours(phB, phA)
    # area-weighted circular RMSE of phase (hours)
    with np.errstate(invalid="ignore"):
        rmse_phase_h = float(np.sqrt(np.nansum(w * dphi_h**2) / np.nansum(w)))
        # amplitude log-ratio RMSE (dimensionless), symmetric
        lnr = np.log(np.divide(ampB, ampA, out=np.ones_like(ampA), where=(ampA>0)))
        rmse_log_amp = float(np.sqrt(np.nansum(w * lnr**2) / np.nansum(w)))

    out = {
        "metric": "|rho|",
        "similarity": similarity,          # PRIMARY SINGLE SCALAR
        "mean_lag_h": mean_lag_h,          # context (B vs A)
        "rmse_phase_h": rmse_phase_h,      # optional diagnostics
        "rmse_log_amp": rmse_log_amp,      # optional diagnostics
        "weights": args.weights,
        "amp_thresh_mm_per_day": args.amp_thresh,
        "var": args.var
    }
    print(json.dumps(out, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()
