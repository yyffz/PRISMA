import xarray as xr
import numpy as np
import pandas as pd
import torch
from datetime import datetime


# Load and subset the NetCDF file
file_path = '/public/home/sunhaofei/cosmos-predict1/workspace/Tokenizer_prep/MESO_20230428_29.nc'
ds = xr.open_dataset(file_path)
print(ds)

# Subset the dataset
ds = ds.sel(time=slice('2023-04-28T00:00', '2023-04-29T08:00'), lat=slice(38.2, 19), lon=slice(97, 127.72))
print(ds)

# Select variables and convert to tensor
vars_selected = ['prmsl', '10u', '10v']
ds_stacked = ds[vars_selected].to_array(dim='channel')  # Shape: (channel:3, time:33, lat:640, lon:1024)
data_np = ds_stacked.values  # Shape: (3, 33, 640, 1024)
print(f"Data shape after stacking: {data_np.shape}")

# Store min and max for each channel before normalization
mins = []
maxs = []
for c in range(data_np.shape[0]):
    min_val = np.nanmin(data_np[c])  # Use nanmin to handle potential NaN values
    max_val = np.nanmax(data_np[c])  # Use nanmax to handle potential NaN values
    print(f"Variable {vars_selected[c]}: min={min_val}, max={max_val}")
    if np.isnan(min_val) or np.isnan(max_val):
        raise ValueError(f"Invalid data detected for {vars_selected[c]}: min={min_val}, max={max_val}")
    mins.append(min_val)
    maxs.append(max_val)
    if max_val > min_val:
        data_np[c] = (data_np[c] - min_val) / (max_val - min_val)
    else:
        print(f"Warning: {vars_selected[c]} has max <= min, skipping normalization")


print(f"mins: {mins}")
print(f"maxs: {maxs}")
# Scale from [0,1] to [-1,1]
data_np = 2 * data_np - 1

# Save the normalized data and scaling parameters to a .npz file
output_file = 'MESO_20230428_29_normalized.npz'
np.savez(output_file, data=data_np, mins=np.array(mins), maxs=np.array(maxs))
print(f"Saved normalized data and scales to {output_file}")