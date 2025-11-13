import argparse
import csv
import os
import time
from typing import Tuple

import numpy as np
import xarray as xr


def detect_coord_name(da: xr.DataArray, candidates: Tuple[str, ...]) -> str:
    for name in candidates:
        if name in da.coords or name in da.dims:
            return name
    raise KeyError(f"Could not find any of coord names {candidates} in data array with dims {da.dims}.")


def ensure_lon_0_360(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    lon = da[lon_name]
    try:
        lon_min = float(lon.min())
    except Exception:
        lon_min = -180.0
    if lon_min < 0.0:
        da = da.assign_coords({lon_name: (lon % 360)})
        da = da.sortby(lon_name)
    return da


def subset_lat_band(da: xr.DataArray, lat_name: str, lat_min: float, lat_max: float) -> xr.DataArray:
    lat_values = da[lat_name].values
    if lat_values[0] <= lat_values[-1]:
        return da.sel({lat_name: slice(lat_min, lat_max)})
    else:
        return da.sel({lat_name: slice(lat_max, lat_min)})


def compute_weighted_mean(da: xr.DataArray, lat_name: str) -> float:
    weights = np.cos(np.deg2rad(da[lat_name]))
    wmean = da.weighted(weights).mean(dim=(lat_name, detect_coord_name(da, ("lon", "longitude"))))
    return float(wmean.values)


def compute_weighted_space_mean_timeseries(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da[lat_name]))
    # Mean over lat/lon, retain time dimension
    return da.weighted(weights).mean(dim=(lat_name, lon_name))


def select_u850(u: xr.DataArray) -> Tuple[xr.DataArray, int, float]:
    lev_name = detect_coord_name(u, ("lev", "plev", "level", "pressure"))
    if lev_name in u.coords:
        # Prefer nearest to 850 hPa
        u850 = u.sel({lev_name: 850}, method="nearest")
        lev_value = float(u850[lev_name].values)
        # Recover index for reporting
        lev_vals = u[lev_name].values
        lev_index = int(np.argmin(np.abs(lev_vals - lev_value)))
        return u850, lev_index, lev_value
    # Fallback to provided index 23 (0-based)
    u850 = u.isel({lev_name: 23})
    lev_value = float(u[lev_name].values[23]) if lev_name in u.coords else np.nan
    return u850, 23, lev_value


def main():
    parser = argparse.ArgumentParser(description="Compute monsoon index from U at 850 hPa.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs.",
    )
    parser.add_argument(
        "--months",
        default="6,7,8",
        help="Comma-separated months (1-12) to include per year, e.g., '6,7,8' for JJA.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Start year (inclusive) for averaging and CSV output.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year (inclusive) for averaging and CSV output.",
    )
    parser.add_argument(
        "--input",
        default="/Users/jiandachen/Projects/e3sm_diags/spcam_climo/U_199901_200312.nc",
        help="Path to input NetCDF file containing U(lev, lat, lon).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to output CSV to append results.",
    )
    args = parser.parse_args()

    months = [int(m.strip()) for m in args.months.split(",") if m.strip()]
    if args.verbose:
        print(f"       Using months: {months}")

    t0 = time.perf_counter()
    if args.verbose:
        print(f"[1/9] Opening dataset: {args.input}")
    t_open0 = time.perf_counter()
    ds = xr.open_dataset(args.input)  # decoding on by default
    t_open1 = time.perf_counter()
    if args.verbose:
        print(f"       Opened dataset in {t_open1 - t_open0:.3f}s")

    if "U" not in ds:
        raise KeyError("Variable 'U' not found in the dataset.")
    u = ds["U"]
    if args.verbose:
        try:
            print(f"       Variable 'U' dims: {u.dims}, shape: {tuple(u.shape)}")
        except Exception:
            pass

    if args.verbose:
        print("[2/9] Detecting coordinate names")
    lat_name = detect_coord_name(u, ("lat", "latitude"))
    lon_name = detect_coord_name(u, ("lon", "longitude"))
    if args.verbose:
        latv = u[lat_name].values
        lonv = u[lon_name].values
        print(f"       Using lat: '{lat_name}' [{float(latv.min()):.3f}, {float(latv.max()):.3f}] "
              f"(ascending={bool(latv[0] <= latv[-1])})")
        try:
            lon_min = float(lonv.min())
            lon_max = float(lonv.max())
            print(f"       Using lon: '{lon_name}' [{lon_min:.3f}, {lon_max:.3f}]")
        except Exception:
            pass

    if args.verbose:
        print("[3/9] Selecting U at ~850 hPa (nearest) or index 23 if needed")
    t_sel0 = time.perf_counter()
    u850, lev_index, lev_value = select_u850(u)
    t_sel1 = time.perf_counter()
    if args.verbose:
        print(f"       Selected level index={lev_index}, value={lev_value:.1f} hPa in {t_sel1 - t_sel0:.3f}s")

    # Normalize longitude to 0–360 and sort
    if args.verbose:
        try:
            lonv = u850[lon_name].values
            print(f"[4/9] Normalizing longitudes to 0–360 (pre-range: {float(lonv.min()):.3f}..{float(lonv.max()):.3f})")
        except Exception:
            print("[4/9] Normalizing longitudes to 0–360")
    u850 = ensure_lon_0_360(u850, lon_name)
    if args.verbose:
        lonv2 = u850[lon_name].values
        print(f"       Post-range: {float(lonv2.min()):.3f}..{float(lonv2.max()):.3f}")

    # Regions
    # A: 10–20N, 100–150E
    # B: 25–35N, 100–150E
    if args.verbose:
        print("[5/9] Subsetting Region A (10–20N, 100–150E)")
    uA = subset_lat_band(u850, lat_name, 10.0, 20.0).sel({lon_name: slice(100.0, 150.0)})
    if args.verbose:
        print("[6/9] Subsetting Region B (25–35N, 100–150E)")
    uB = subset_lat_band(u850, lat_name, 25.0, 35.0).sel({lon_name: slice(100.0, 150.0)})

    # Branch on time dimension: per-year computation if time present
    time_name = None
    for cand in ("time", "Time"):
        if cand in uA.dims or cand in uA.coords:
            time_name = cand
            break

    if time_name is not None:
        if args.verbose:
            print("[7/9] Filtering months and computing per-year indices")
        # Filter selected months
        month_mask = uA[time_name].dt.month.isin(months)
        uA_sel = uA.sel({time_name: month_mask})
        uB_sel = uB.sel({time_name: month_mask})

        # Space-mean time series
        series_A = compute_weighted_space_mean_timeseries(uA_sel, lat_name, lon_name)
        series_B = compute_weighted_space_mean_timeseries(uB_sel, lat_name, lon_name)

        # Determine if months wrap across year boundary (e.g., DJF: 12,1,2)
        wraps_year = (12 in months) and any(m <= 2 for m in months)
        if args.verbose:
            if wraps_year:
                print("       Detected cross-year months; assigning December to next year's season.")
            else:
                print("       No cross-year wrap; grouping by calendar year.")

        # Group by season year: if wraps_year, December belongs to next year
        if wraps_year:
            season_year = series_A[time_name].dt.year + xr.where(series_A[time_name].dt.month == 12, 1, 0)
            mean_A_year = series_A.groupby(season_year).mean()
            mean_B_year = series_B.groupby(season_year).mean()
        else:
            mean_A_year = series_A.groupby(f"{time_name}.year").mean()
            mean_B_year = series_B.groupby(f"{time_name}.year").mean()

        # Compute yearly differences
        diff_year = mean_A_year - mean_B_year

        # Extract year labels regardless of grouping method
        group_dim = list(mean_A_year.dims)[0]
        years = mean_A_year[group_dim].values

        # Apply year filtering
        years_np = np.array(years, dtype=int)
        mask = np.ones_like(years_np, dtype=bool)
        if args.start_year is not None:
            mask &= years_np >= args.start_year
        if args.end_year is not None:
            mask &= years_np <= args.end_year
        if args.verbose:
            if args.start_year is not None or args.end_year is not None:
                print(f"       Applying year filter: "
                      f"{args.start_year if args.start_year is not None else years_np.min()} - "
                      f"{args.end_year if args.end_year is not None else years_np.max()}")
        years_f = years_np[mask]
        mean_A_vals = mean_A_year.values[mask]
        mean_B_vals = mean_B_year.values[mask]
        diff_vals = (diff_year.values[mask])

        if args.verbose:
            print("[8/9] Computing yearly differences (A - B)")
        # diff_year already computed; filtered above as diff_vals

        print(f"File: {args.input}")
        print(f"Selected level index: {lev_index}, level value: {lev_value:.1f} hPa")
        for i, year in enumerate(years_f):
            a_val = float(mean_A_vals[i])
            b_val = float(mean_B_vals[i])
            d_val = float(diff_vals[i])
            print(f"Year {int(year)}: A={a_val:.4f} m/s, B={b_val:.4f} m/s, A-B={d_val:.4f} m/s")

        # Final multi-year mean of the yearly monsoon index
        if years_f.size > 0:
            final_mean = float(np.nanmean(diff_vals))
            yr_min = int(years_f.min())
            yr_max = int(years_f.max())
            print(f"Final mean monsoon index over years {yr_min}-{yr_max}: {final_mean:.4f} m/s")
        else:
            print("No years remaining after filter; final mean not computed.")

        # Write CSV rows per year
        if args.verbose:
            print("[9/9] Writing yearly results to CSV")
        if args.output_csv:
            out_dir = os.path.dirname(args.output_csv)
            os.makedirs(out_dir, exist_ok=True)
            write_header = not (os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0)
            with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        ["file", "year", "months", "lev_index", "lev_value_hPa", "mean_A", "mean_B", "diff_A_minus_B"]
                    )
                for i, year in enumerate(years_f):
                    writer.writerow([
                        args.input,
                        int(year),
                        ",".join(str(m) for m in months),
                        lev_index,
                        f"{lev_value:.1f}",
                        f"{float(mean_A_vals[i]):.6f}",
                        f"{float(mean_B_vals[i]):.6f}",
                        f"{float(diff_vals[i]):.6f}",
                    ])
        if args.verbose:
            t1 = time.perf_counter()
            print(f"Done in {t1 - t0:.3f}s")
    else:
        if args.verbose:
            print("[7/9] Computing cosine-weighted means (no time dimension)")
        t_mean0 = time.perf_counter()
        mean_A = compute_weighted_mean(uA, lat_name)
        mean_B = compute_weighted_mean(uB, lat_name)
        t_mean1 = time.perf_counter()
        if args.verbose:
            print(f"       Means computed in {t_mean1 - t_mean0:.3f}s")

        if args.verbose:
            print("[8/9] Computing difference (A - B)")
        diff = mean_A - mean_B

        print(f"File: {args.input}")
        print(f"Selected level index: {lev_index}, level value: {lev_value:.1f} hPa")
        print(f"Region A (10–20N, 100–150E) U850 mean: {mean_A:.4f} m/s")
        print(f"Region B (25–35N, 100–150E) U850 mean: {mean_B:.4f} m/s")
        print(f"Difference (A - B): {diff:.4f} m/s")

        # Write CSV
        if args.verbose:
            print("[9/9] Writing results to CSV")
        out_dir = os.path.dirname(args.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        write_header = not (os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0)
        with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["file", "lev_index", "lev_value_hPa", "mean_A", "mean_B", "diff_A_minus_B"]
                )
            writer.writerow([args.input, lev_index, f"{lev_value:.1f}", f"{mean_A:.6f}", f"{mean_B:.6f}", f"{diff:.6f}"])
        if args.verbose:
            t1 = time.perf_counter()
            print(f"Done in {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()


