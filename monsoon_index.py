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


def select_u_at_pressure(u: xr.DataArray, target_pressure_hpa: float) -> Tuple[xr.DataArray, int, float]:
    lev_name = detect_coord_name(u, ("lev", "plev", "level", "pressure"))
    if lev_name in u.coords:
        # Prefer nearest to requested pressure (assumed in hPa)
        u_sel = u.sel({lev_name: target_pressure_hpa}, method="nearest")
        lev_value = float(u_sel[lev_name].values)
        # Recover index for reporting
        lev_vals = u[lev_name].values
        lev_index = int(np.argmin(np.abs(lev_vals - lev_value)))
        return u_sel, lev_index, lev_value
    # Fallback only for known 850 hPa index if no coordinate exists
    if int(target_pressure_hpa) == 850:
        u_sel = u.isel({lev_name: 23})
        lev_value = float(u[lev_name].values[23]) if lev_name in u.coords else np.nan
        return u_sel, 23, lev_value
    raise ValueError("Level coordinate not found; cannot select target pressure without coordinates.")


def main():
    parser = argparse.ArgumentParser(description="Compute monsoon index from U at 850 hPa.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs.",
    )
    parser.add_argument(
        "--season",
        choices=["JJA", "DJF"],
        default=None,
        help="Season configuration to use: JJA (summer) or DJF (winter). If omitted, both will be computed.",
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

    # Dataset open and coord detection are shared across seasons

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

    # If no season specified, compute both JJA and DJF sequentially and return
    if args.season is None:
        for season in ("JJA", "DJF"):
            months = [6, 7, 8] if season == "JJA" else [12, 1, 2]
            target_pressure = 850 if season == "JJA" else 500
            if args.verbose:
                print(f"\n=== Season {season} ===")
                print(f"Using months={months}; target pressure={target_pressure} hPa")

            if args.verbose:
                print(f"[3/9] Selecting U at ~{target_pressure} hPa (nearest)")
            t_sel0 = time.perf_counter()
            u_sel, lev_index, lev_value = select_u_at_pressure(u, target_pressure)
            t_sel1 = time.perf_counter()
            if args.verbose:
                print(f"       Selected level index={lev_index}, value={lev_value:.1f} hPa in {t_sel1 - t_sel0:.3f}s")

            # Normalize longitude to 0–360 and sort
            if args.verbose:
                try:
                    lonv = u_sel[lon_name].values
                    print(f"[4/9] Normalizing longitudes to 0-360 (pre-range: {float(lonv.min()):.3f}..{float(lonv.max()):.3f})")
                except Exception:
                    print("[4/9] Normalizing longitudes to 0-360")
            u_sel = ensure_lon_0_360(u_sel, lon_name)
            if args.verbose:
                lonv2 = u_sel[lon_name].values
                print(f"       Post-range: {float(lonv2.min()):.3f}..{float(lonv2.max()):.3f}")

            # Regions by season
            if args.verbose:
                print("[5/9] Subsetting region(s)")
            if season == "JJA":
                uA = subset_lat_band(u_sel, lat_name, 10.0, 20.0).sel({lon_name: slice(100.0, 150.0)})
                if args.verbose:
                    print("       Region A (10-20N, 100-150E)")
                uB = subset_lat_band(u_sel, lat_name, 25.0, 35.0).sel({lon_name: slice(100.0, 150.0)})
                if args.verbose:
                    print("       Region B (25-35N, 100-150E)")
            else:
                uA = subset_lat_band(u_sel, lat_name, 25.0, 35.0).sel({lon_name: slice(80.0, 120.0)})
                uB = subset_lat_band(u_sel, lat_name, 50.0, 60.0).sel({lon_name: slice(80.0, 120.0)})
                if args.verbose:
                    print("       Region A (25-35N, 80-120E)")
                    print("       Region B (50-60N, 80-120E)")

            # Per-year computation
            time_name = None
            for cand in ("time", "Time"):
                if cand in uA.dims or cand in uA.coords:
                    time_name = cand
                    break

            if time_name is None:
                # No time dimension; compute single values
                if args.verbose:
                    print("[7/9] Computing cosine-weighted means (no time dimension)")
                mean_A = compute_weighted_mean(uA, lat_name)
                mean_B = compute_weighted_mean(uB, lat_name)
                diff = mean_A - mean_B
                print(f"File: {args.input}")
                print(f"Season {season} | Level index: {lev_index}, value: {lev_value:.1f} hPa")
                print(f"A mean: {mean_A:.4f} m/s, B mean: {mean_B:.4f} m/s, A-B: {diff:.4f} m/s")
                continue

            if args.verbose:
                print("[7/9] Filtering months and computing per-year indices")
            month_mask = uA[time_name].dt.month.isin(months)
            uA_sel = uA.sel({time_name: month_mask})
            uB_sel = uB.sel({time_name: month_mask})

            series_A = compute_weighted_space_mean_timeseries(uA_sel, lat_name, lon_name)
            series_B = compute_weighted_space_mean_timeseries(uB_sel, lat_name, lon_name)

            wraps_year = (12 in months) and any(m <= 2 for m in months)
            if args.verbose:
                if wraps_year:
                    print("       Detected cross-year months; assigning December to next year's season.")
                else:
                    print("       No cross-year wrap; grouping by calendar year.")

            if wraps_year:
                season_year = series_A[time_name].dt.year + xr.where(series_A[time_name].dt.month == 12, 1, 0)
                mean_A_year = series_A.groupby(season_year).mean()
                mean_B_year = series_B.groupby(season_year).mean()
            else:
                mean_A_year = series_A.groupby(f"{time_name}.year").mean()
                mean_B_year = series_B.groupby(f"{time_name}.year").mean()
            diff_year = mean_A_year - mean_B_year
            group_dim = list(mean_A_year.dims)[0]
            years = mean_A_year[group_dim].values

            years_np = np.array(years, dtype=int)
            mask = np.ones_like(years_np, dtype=bool)
            if args.start_year is not None:
                mask &= years_np >= args.start_year
            if args.end_year is not None:
                mask &= years_np <= args.end_year
            years_f = years_np[mask]
            mean_A_vals = mean_A_year.values[mask]
            mean_B_vals = mean_B_year.values[mask]
            diff_vals = diff_year.values[mask]

            print(f"File: {args.input}")
            print(f"Season {season} | Selected level index: {lev_index}, level value: {lev_value:.1f} hPa")
            for i, year in enumerate(years_f):
                a_val = float(mean_A_vals[i])
                b_val = float(mean_B_vals[i])
                d_val = float(diff_vals[i])
                print(f"{season} Year {int(year)}: A={a_val:.4f} m/s, B={b_val:.4f} m/s, A-B={d_val:.4f} m/s")

            if years_f.size > 0:
                final_mean = float(np.nanmean(diff_vals))
                yr_min = int(years_f.min())
                yr_max = int(years_f.max())
                print(f"{season} final mean monsoon index over years {yr_min}-{yr_max}: {final_mean:.4f} m/s")
            else:
                print(f"{season}: No years remaining after filter; final mean not computed.")

            if args.output_csv is not None:
                if args.verbose:
                    print("[9/9] Writing yearly results to CSV")
                out_dir = os.path.dirname(args.output_csv)
                if out_dir:
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
                            season,
                            lev_index,
                            f"{lev_value:.1f}",
                            f"{float(mean_A_vals[i]):.6f}",
                            f"{float(mean_B_vals[i]):.6f}",
                            f"{float(diff_vals[i]):.6f}",
                        ])
        return
    # Single-season path continues here
    if args.season == "JJA":
        months = [6, 7, 8]
        target_pressure = 850
    else:
        months = [12, 1, 2]
        target_pressure = 500
    if args.verbose:
        print(f"       Using season: {args.season}; months={months}; target pressure={target_pressure} hPa")
        print("[3/9] Selecting U at ~{target_pressure} hPa (nearest)")
    t_sel0 = time.perf_counter()
    u_sel, lev_index, lev_value = select_u_at_pressure(u, target_pressure)
    t_sel1 = time.perf_counter()
    if args.verbose:
        print(f"       Selected level index={lev_index}, value={lev_value:.1f} hPa in {t_sel1 - t_sel0:.3f}s")

    # Normalize longitude to 0–360 and sort
    if args.verbose:
        try:
            lonv = u_sel[lon_name].values
            print(f"[4/9] Normalizing longitudes to 0-360 (pre-range: {float(lonv.min()):.3f}..{float(lonv.max()):.3f})")
        except Exception:
            print("[4/9] Normalizing longitudes to 0-360")
    u_sel = ensure_lon_0_360(u_sel, lon_name)
    if args.verbose:
        lonv2 = u_sel[lon_name].values
        print(f"       Post-range: {float(lonv2.min()):.3f}..{float(lonv2.max()):.3f}")

    # Regions
    # JJA:
    #   A: 10–20N, 100–150E
    #   B: 25–35N, 100–150E
    # DJF:
    #   A: 25–35N, 80–120E
    #   B: 50–60N, 80–120E
    if args.verbose:
        print("[5/9] Subsetting region(s)")
    if args.season == "JJA":
        uA = subset_lat_band(u_sel, lat_name, 10.0, 20.0).sel({lon_name: slice(100.0, 150.0)})
        if args.verbose:
            print("       Region A (10-20N, 100-150E)")
        uB = subset_lat_band(u_sel, lat_name, 25.0, 35.0).sel({lon_name: slice(100.0, 150.0)})
        if args.verbose:
            print("       Region B (25-35N, 100-150E)")
    else:
        uA = subset_lat_band(u_sel, lat_name, 25.0, 35.0).sel({lon_name: slice(80.0, 120.0)})
        uB = subset_lat_band(u_sel, lat_name, 50.0, 60.0).sel({lon_name: slice(80.0, 120.0)})
        if args.verbose:
            print("       Region A (25-35N, 80-120E)")
            print("       Region B (50-60N, 80-120E)")

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
        if args.season == "JJA":
            uB_sel = uB.sel({time_name: month_mask})
        else:
            uB_sel = uB.sel({time_name: month_mask})

        # Space-mean time series
        series_A = compute_weighted_space_mean_timeseries(uA_sel, lat_name, lon_name)
        if args.season == "JJA":
            series_B = compute_weighted_space_mean_timeseries(uB_sel, lat_name, lon_name)
        else:
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
        diff_year = mean_A_year - mean_B_year
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
            print("[8/9] Computing yearly monsoon index")
        # Values already computed above; filtered as diff_vals or index_vals

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
        if args.output_csv is not None:
            if args.verbose:
                print("[9/9] Writing yearly results to CSV")
            out_dir = os.path.dirname(args.output_csv)
            if out_dir:
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
                        args.season,
                        lev_index,
                        f"{lev_value:.1f}",
                        f"{float(mean_A_vals[i]):.6f}",
                        f"{float(mean_B_vals[i]):.6f}",
                        f"{float(diff_vals[i]):.6f}",
                    ])
            if args.verbose:
                t1 = time.perf_counter()
                print(f"Done in {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()


