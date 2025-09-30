import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import cftime
import glob
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

#def process_file(file_path):
#    """Process a single NetCDF file and return hourly data."""
#    print(f"\nProcessing file: {os.path.basename(file_path)}")
#
#    # Open dataset with chunking
#    ds = xr.open_dataset(file_path, chunks={'time': 24},
#                        engine='netcdf4',
#                        decode_times=True)
#
#    # Select only the required variables
#    required_vars = ['time', 'date', 'datesec', 'time_bnds', 'SPPRECC', 'FLUT', 'UL']
#    ds = ds[required_vars]
    # Extract UL at level 23 (850hPa) and rename it to U850
##    #ds['U850'] = ds.UL.isel(lev=23)
#    ds = ds.drop_vars('UL')
#    # Rename SPPRECC to PRECT
#    ds = ds.rename({'SPPRECC': 'PRECT'})import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import cftime
import glob
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    """Process a single NetCDF file and return daily mean data."""
    print(f"\n📂 Processing file: {os.path.basename(file_path)}")

    ds = xr.open_dataset(file_path, chunks={'time': 24},
                         engine='netcdf4', decode_times=True)

    required_vars = ['time', 'date', 'datesec', 'time_bnds', 'cp', 'FLUT', 'U850']
    ds = ds[required_vars]

    # Extract UL at level 23 (850hPa) and rename
    #ds['U850'] = ds["U
    #ds = ds.drop_vars('UL')

    # Rename SPPRECC to PRECT
    ds = ds.rename({'cp': 'PRECT'})

    # Resample to daily mean
    daily_ds = ds.resample(time='1D').mean()

    # Update time fields
    daily_ds['date'] = daily_ds.time.dt.strftime('%Y%m%d').astype(int)
    daily_ds['datesec'] = 43200  # midpoint of the day (12:00:00)

    # Create time bounds: daily ±12 hours
    time_values = daily_ds.time.values
    time_bnds = np.empty((len(time_values), 2), dtype='datetime64[ns]')

    for i, t in enumerate(time_values):
        time_bnds[i, 0] = t - np.timedelta64(12, 'h')
        time_bnds[i, 1] = t + np.timedelta64(12, 'h')

    daily_ds['time_bnds'] = xr.DataArray(
        time_bnds,
        dims=['time', 'nbnd'],
        coords={'time': daily_ds['time'], 'nbnd': [0, 1]},
        name='time_bnds'
    )

    daily_ds['time'].attrs['bounds'] = 'time_bnds'

    # Restore attributes
    for var in daily_ds.data_vars:
        if var in ds.data_vars:
            daily_ds[var].attrs = ds[var].attrs

    ds.close()
    return daily_ds


def main():
    for year in range(1998, 2004):
        #file_pattern = f'/share1/x-w19/raw-spcam-data/spcam_std_rad.cam.h1.{year}-*.nc'
        file_pattern = f'/pscratch/sd/c/chenjd21/online_simulation_nc_data/er_mix1.0_std/conv_mem_spinup5.cam.h1.{year}-01-01-00000.nc'
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

        combined_ds.attrs = {
            'title': f'Daily mean of time and precipitation data from SP-CAM for {year}',
            'source_files': [os.path.basename(f) for f in file_list],
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_note': 'Data resampled to daily frequency using mean values and combined for the entire year'
        }

        output_file = f'er_daily_mean_time_precip_{year}.nc'
        print(f"\n💾 Saving combined dataset to: {output_file}")
        combined_ds.to_netcdf(output_file)

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
    main()

