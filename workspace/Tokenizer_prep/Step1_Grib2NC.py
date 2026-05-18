import pygrib
import xarray as xr
import numpy as np
import os
from datetime import datetime
import glob


# file_path = './MESO_20230428/postvar202304280000100.grib'
# grbs = pygrib.open(file_path)
# print("All GRIB messages:")
# for grb in grbs:
#     print(f"Message: {grb.messagenumber}, Name: {grb.name}, ShortName: {grb.shortName}, ParamId: {grb.paramId}, Type of Level: {grb.typeOfLevel}, Level: {grb.level}")
# msg = grbs[1]
# lats, lons = msg.latlons()
# print(f"range of latitudes: {np.min(lats)} to {np.max(lats)}")
# print(f"range of longitudes: {np.min(lons)} to {np.max(lons)}")
# print(f"Latitude shape: {lats.shape}, Longitude shape: {lons.shape}")


file_dir = './MESO_20230428/'
file_list = sorted(glob.glob(os.path.join(file_dir, 'postvar*.grib')))
data_vars = {
    'prmsl': [],
    '2t': [],
    '10u': [],
    '10v': []
}
times = []
lats, lons = None, None

# Process each GRIB file
for file_path in file_list:
    file_name = os.path.basename(file_path)
    time_str = file_name[7:17]  # Extract YYYYMMDDHHMM
    print(f"Processing file: {file_name}, time_str: {time_str}")
    time = datetime.strptime(time_str, '%Y%m%d%H')
    times.append(time)
    

    grbs = pygrib.open(file_path)
    temp_vars = {}
    for grb in grbs:
        if grb.messagenumber == 288 and grb.shortName == 'prmsl':
            temp_vars['prmsl'] = grb.values
            lats, lons = grb.latlons()
        elif grb.messagenumber == 297 and grb.shortName == '2t':
            temp_vars['2t'] = grb.values - 273.15  # Convert Kelvin to Celsius
        elif grb.messagenumber == 298 and grb.shortName == '10u':
            temp_vars['10u'] = grb.values
        elif grb.messagenumber == 299 and grb.shortName == '10v':
            temp_vars['10v'] = grb.values
    
    grbs.close()
    for var in data_vars:
        data_vars[var].append(temp_vars[var])

# Convert lists to numpy arrays with time dimension
for var in data_vars:
    data_vars[var] = np.stack(data_vars[var], axis=0)

# Create xarray.Dataset
ds = xr.Dataset(
    {
        'prmsl': (['time', 'lat', 'lon'], data_vars['prmsl']),
        '2t': (['time', 'lat', 'lon'], data_vars['2t']),
        '10u': (['time', 'lat', 'lon'], data_vars['10u']),
        '10v': (['time', 'lat', 'lon'], data_vars['10v']),
    },
    coords={
        'time': times,
        'lat': (['lat'], lats[:, 0]),
        'lon': (['lon'], lons[0, :]),
    }
)

# Print dataset to verify
print(ds)
# save to NetCDF file
output_file = 'MESO_20230428_29.nc'
ds.to_netcdf(output_file)
print(f"Dataset saved to {output_file}")    