import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import cftime
import glob
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    print(f"\n📂 Processing file: {os.path.basename(file_path)}")

    ds = xr.open_dataset(
        file_path,
        chunks={'time': 24},
        engine='netcdf4',
        decode_times=True,
        use_cftime=True  # 👈 强制使用 cftime 时间
    )

    #required_vars = ['time', 'date', 'datesec', 'time_bnds', 'PRECT', 'FLUT', 'U850']
    required_vars = ['time', 'date', 'datesec', 'time_bnds', 'cp', 'FLUT', 'U850', 'TMQ']
    #required_vars = ['time', 'date', 'datesec', 'time_bnds', 'PRECT', 'FLUT', 'U850', 'TMQ']
    #required_vars = ['time', 'date', 'datesec', 'time_bnds', 'PRECT', 'FLUT', 'U850', 'TMQ']
    ds = ds[required_vars]

    # 提取 UL @ lev=23 -> U850
    #ds['U850'] = ds.UL.isel(lev=23)
    #ds = ds.drop_vars('UL')

    # 改名
    ds = ds.rename({'cp': 'PRECT'})

    # 重采样为每日平均
    daily_ds = ds.resample(time='1D').mean()

    # 设置 date, datesec
    daily_ds['date'] = xr.DataArray(
        [int(f"{t.year:04}{t.month:02}{t.day:02}") for t in daily_ds.time.values],
        dims='time'
    )
    daily_ds['datesec'] = xr.DataArray(
        [12 * 3600 for _ in daily_ds.time.values],  # 中午12点
        dims='time'
    )

    # 设置 time_bnds（使用 datetime.timedelta 替换 np.timedelta64）
    time_bnds = np.empty((len(daily_ds.time), 2), dtype=object)
    for i, t in enumerate(daily_ds.time.values):
        time_bnds[i, 0] = t - timedelta(hours=12)
        time_bnds[i, 1] = t + timedelta(hours=12)

    daily_ds['time_bnds'] = xr.DataArray(
        time_bnds,
        dims=['time', 'nbnd'],
        coords={'time': daily_ds.time, 'nbnd': [0, 1]},
        name='time_bnds'
    )
    daily_ds['time'].attrs['bounds'] = 'time_bnds'

    # 拷贝原始变量属性
    for var in daily_ds.data_vars:
        if var in ds.data_vars:
            daily_ds[var].attrs = ds[var].attrs

    ds.close()
    return daily_ds

def main(case_dir, case_name):
    for year in range(1998,1999):
        #file_pattern = f'/share1/x-w19/raw-spcam-data/spcam_std_rad.cam.h1.{year}-*.nc'
        #file_pattern = f'/share3/chenj209/raw_spcam_data/spcam_std_rad.cam.h1.{year}-*.nc'
        #file_pattern = f'/pscratch/sd/c/chenjd21/online_simulation_nc_data/er_mix1.0_std/conv_mem_spinup5.cam.h1.{year}-01-01-00000.nc'
        #file_pattern = f'/pscratch/sd/c/chenjd21/20250818.CAM5_F_2000_CAM5_f19_g16/archive/atm/hist/20250818.CAM5_F_2000_CAM5_f19_g16.cam.h1.{year}-*.nc'
        #file_pattern = f'/pscratch/sd/c/chenjd21/online_simulation_nc_data/nncam/conv_mem_share3.cam.h1.{year}-*.nc'
        #file_pattern = f'/pscratch/sd/c/chenjd21/NNCAM_INTEL_20250710_1114_F_2000_SPCAM_m2005_f19_g16/run/NNCAM_INTEL_20250710_1114_F_2000_SPCAM_m2005_f19_g16.cam.h1.{year}-*.nc'
        file_pattern = f'{case_dir}/{case_name}.cam.h1.{year}-*.nc'
        file_list = sorted(glob.glob(file_pattern))

        if not file_list:
            print(f"No files found matching pattern: {file_pattern}")
            return

        print(f"📅 Found {len(file_list)} files to process")

        daily_datasets = []
        for file_path in tqdm(file_list):
            try:
                daily_ds = process_file(file_path)
                daily_datasets.append(daily_ds)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")

        if not daily_datasets:
            print("⚠️ No datasets were successfully processed")
            return

        print("\n🔗 Combining daily datasets...")
        combined_ds = xr.concat(daily_datasets, dim='time')
        combined_ds = combined_ds.sortby('time')

        # Add global attributes
        combined_ds.attrs = {
            'title': f'Daily mean of time and precipitation data from SP-CAM for {year}',
            'source_files': [os.path.basename(f) for f in file_list],
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_note': 'Data resampled to daily frequency using mean/sum values and combined for the entire year'
        }

        # 🩹 FIX CF-compliance warning
        if 'units' not in combined_ds.time.encoding:
            combined_ds.time.encoding['units'] = 'days since 1990-01-01 00:00:00'
        if 'calendar' not in combined_ds.time.encoding:
            combined_ds.time.encoding['calendar'] = 'noleap'

        combined_ds.time_bnds.encoding['units'] = combined_ds.time.encoding['units']
        combined_ds.time_bnds.encoding['calendar'] = combined_ds.time.encoding['calendar']

        #output_file = f'cam_daily_mean_time_precip_{year}.nc'
        #output_file = f'20250818.CAM5_F_2000_CAM5_f19_g16.cam.h2.{year}-01.nc'
        #output_file = f'conv_mem_share3.cam.h2.{year}-01.nc'
        output_file = f'{case_name}.cam.pw.{year}-01.nc'
        print(f"\n💾 Saving combined dataset to: {output_file}")
        combined_ds.to_netcdf(output_file)

        from netCDF4 import Dataset

        filename = output_file

        with Dataset(filename, mode="r+") as nc:
            # Confirm the variable exists
            if "time_bnds" in nc.variables:
                time_bnds = nc.variables["time_bnds"]
                time_bnds.units = "days since 1990-01-01 00:00:00"
                time_bnds.calendar = "noleap"
                print("✅ Successfully added 'units' and 'calendar' to time_bnds.")
            else:
                print("❌ Variable 'time_bnds' not found.")

        print("\n=== ✅ Verification Information ===")
        print(f"Total time steps: {len(combined_ds.time)}")
        print(f"Time range: {combined_ds.time.values[0]} to {combined_ds.time.values[-1]}")
        print(f"Time frequency: {combined_ds.time.diff('time').values[0]}")

        print("\nPRECT statistics:")
        print(f"Mean: {np.mean(combined_ds.PRECT.values):.6f}")
        print(f"Std: {np.std(combined_ds.PRECT.values):.6f}")
        print(f"Min: {np.min(combined_ds.PRECT.values):.6f}")
        print(f"Max: {np.max(combined_ds.PRECT.values):.6f}")

        print("\nSpatial dimensions:")
        print(f"Lat/lon shape: {combined_ds.PRECT.shape[1:]}")

        print(f"\nNaN count: {np.isnan(combined_ds.PRECT.values).sum()}")

        for ds in daily_datasets:
            ds.close()
        combined_ds.close()

        print(f"\n✅ Successfully created daily dataset: {output_file}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
