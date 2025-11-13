#!/usr/bin/env python3
"""
annual_cycle_skill_from_monthly.py

Compute a single-scalar similarity between two sets of monthly climo files
(test vs. reference) for a variable such as PRECT.

Inputs:
  --var        variable name (e.g. PRECT)
  --test_dir   directory containing 01–12 monthly *_climo.nc files (test)
  --ref_dir    same for reference
Output:
  JSON with skill, correlation, RMSE, NRMSE, std_ratio
"""

import argparse, glob, os, json
import numpy as np
import xarray as xr

def coslat_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return xr.DataArray(np.clip(w, 0, None), coords={"lat": lat}, dims="lat")

import re

def load_monthly_climos(path_dir, var):
    # Match files that contain _MM_ where MM = 01..12
    files = sorted(glob.glob(os.path.join(path_dir, "*_[0-1][0-9]_*.nc")))
    if len(files) == 0:
        raise FileNotFoundError(f"No monthly climo files found in {path_dir}")

    fields = []
    for f in files:
        # extract month number using regex
        match = re.search(r"_([0-1][0-9])_", os.path.basename(f))
        if not match:
            continue
        m = int(match.group(1))
        ds = xr.open_dataset(f)
        da = ds[var].squeeze()  # (lat, lon)
        if getattr(da, "units", "").lower() in ["m/s", "m s-1", "m s^-1"]:
            da = da * 86400.0 * 1000.0  # convert to mm/day
        if "lon" in da.dims:
            da = da.mean("lon")  # zonal mean
        da = da.assign_coords(month=m)
        fields.append(da)
    if len(fields) == 0:
        raise ValueError(f"No valid monthly files parsed in {path_dir}")
    fields = xr.concat(fields, dim="month").sortby("month")
    return fields  # (month, lat)

def weighted_stats(test, ref, wlat):
    w = wlat.broadcast_like(test)
    w = xr.where(np.isfinite(test) & np.isfinite(ref), w, 0.0)

    def wmean(x): return (w*x).sum(("lat","month")) / w.sum(("lat","month"))
    t_mean, r_mean = wmean(test), wmean(ref)
    t_anom, r_anom = test - t_mean, ref - r_mean

    def wvar(xa): return (w*xa*xa).sum(("lat","month")) / w.sum(("lat","month"))
    var_t, var_r = wvar(t_anom), wvar(r_anom)
    cov = (w*t_anom*r_anom).sum(("lat","month")) / w.sum(("lat","month"))
    r = (cov / np.sqrt(var_t*var_r)).item() if (var_t>0 and var_r>0) else np.nan

    diff = test - ref
    mse = (w*diff*diff).sum(("lat","month")) / w.sum(("lat","month"))
    rmse = float(np.sqrt(mse))
    std_ref = float(np.sqrt(var_r))
    nrmse = rmse / std_ref if std_ref>0 else np.nan
    std_ratio = float(np.sqrt(var_t)/std_ref) if std_ref>0 else np.nan
    return r, rmse, nrmse, std_ratio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    test = load_monthly_climos(args.test_dir, args.var)
    ref  = load_monthly_climos(args.ref_dir,  args.var)
    test, ref = xr.align(test, ref, join="inner")
    wlat = coslat_weights(test.lat)

    r, rmse, nrmse, std_ratio = weighted_stats(test, ref, wlat)
    skill = float(r * np.exp(-(nrmse**2))) if np.isfinite(r) and np.isfinite(nrmse) else np.nan

    print(json.dumps({
        "metric": "annual_zonal_cycle_skill",
        "var": args.var,
        "skill": skill,
        "pattern_correlation": r,
        "rmse_mm_day": rmse,
        "nrmse": nrmse,
        "std_ratio": std_ratio
    }, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()
