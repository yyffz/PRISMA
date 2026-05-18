import xarray as xr
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.mpl.ticker as cticker
from matplotlib.colors import BoundaryNorm


# Function to set up plot axes
# Define plotting function with cartopy
def PlotAxis(ax):
    bwith = 1.5
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)

    # Set extent to subsetted region
    WestLon, EastLon, SouthLat, NorthLat = [97, 127.72, 19, 38.2]
    Extent = [WestLon, EastLon, SouthLat, NorthLat]
    ax.set_extent(Extent, crs=ccrs.PlateCarree())

    ax.set_xticks(np.arange(100, 128, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(19, 39, 2), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=7)
    return None


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

# # Save the normalized data and scaling parameters to a .npz file
# output_file = 'MESO_20230428_29_normalized.npz'
# np.savez(output_file, data=data_np, mins=np.array(mins), maxs=np.array(maxs))
# print(f"Saved normalized data and scales to {output_file}")

# predict results
predict = np.load("../../assets/autoregressive/autoregressive-4b_MESO_20230428.npz")
print(predict.files)
data_predict = predict['data']  # Shape: (3, 33, 640, 1024)
# transpose from (0,1) to original range


# Reverse normalization: from [0,1] to  original range
reconstructed_np = np.zeros_like(data_predict)
for c in range(data_predict.shape[0]):
    if maxs[c] > mins[c]:
        reconstructed_np[c] = data_predict[c] * (maxs[c] - mins[c]) + mins[c]
    else:
        print(f"Warning: {vars_selected[c]} has max <= min, using raw predicted data")
        reconstructed_np[c] = data_predict[c]

# Define levels for each variable based on original ranges
levels = {}
for c, var in enumerate(vars_selected):
    min_val = min(mins[c], maxs[c])  # Ensure min is smaller
    max_val = max(mins[c], maxs[c])  # Ensure max is larger
    if var == 'prmsl' and max_val > 10000:  # Assume Pa if values are large, convert to hPa
        min_val = min_val / 100
        max_val = max_val / 100
        levels[var] = np.linspace(np.round(min_val), np.round(max_val), 20).astype(int)
    else:
        levels[var] = np.linspace(min_val, max_val, 20)
    print(f"Contour levels for {var}: {levels[var]}")





# Loop over each time step to create separate images
for t in range(33):
    # Set up figure for this time step
    fig, axs = plt.subplots(2, 3, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.tight_layout(pad=5.0, h_pad=8.0, rect=[0, 0.1, 1, 0.95])  # Larger row gap, reserve space at bottom

    time_str = pd.Timestamp(ds.time.values[t]).strftime('%Y-%m-%d %H:%M')
    print(f"Processing time: {time_str}")

    for c, var in enumerate(vars_selected):
        print(f"Processing variable: {var}, channel: {c}, time step: {t}")
        print(f"Original data range for {var}: min={mins[c]}, max={maxs[c]}")
        print(
            f"Reconstructed data range for {var}: min={np.nanmin(reconstructed_np[c, t])}, max={np.nanmax(reconstructed_np[c, t])}")

        # Prepare data for plotting (handle units for prmsl)
        original_data = ds[var].isel(time=t)
        recon_data = reconstructed_np[c, t]

        # Plot original data
        PlotAxis(axs[0, c])
        norm = BoundaryNorm(levels[var], ncolors=plt.cm.jet.N, clip=True)
        contour_orig = axs[0, c].contourf(ds.lon, ds.lat, original_data, levels=levels[var],
                                          cmap='jet', norm=norm, extend='both', transform=ccrs.PlateCarree())
        axs[0, c].gridlines(linewidth=0.6, color='black', alpha=0.5, linestyle='--')
        shp_file_path = '/public/home/sunhaofei/plot_use/china1.shp'
        shp = shpreader.Reader(shp_file_path).geometries()
        axs[0, c].add_geometries(shp, ccrs.PlateCarree(), facecolor='None', edgecolor='black', zorder=1, lw=0.6,
                                 alpha=0.4)
        axs[0, c].set_title(f'Original {var} {time_str} UTC', fontsize=9)

        # Plot reconstructed data
        PlotAxis(axs[1, c])
        contour_recon = axs[1, c].contourf(ds.lon, ds.lat, recon_data, levels=levels[var],
                                           cmap='jet', norm=norm, extend='both', transform=ccrs.PlateCarree())
        axs[1, c].gridlines(linewidth=0.6, color='black', alpha=0.5, linestyle='--')
        shp_file_path = '/public/home/sunhaofei/plot_use/china1.shp'
        shp = shpreader.Reader(shp_file_path).geometries()
        axs[1, c].add_geometries(shp, ccrs.PlateCarree(), facecolor='None', edgecolor='black', zorder=1, lw=0.6,
                                 alpha=0.4)
        axs[1, c].set_title(f'Predict {var} {time_str} UTC', fontsize=9)

        # Shared colorbar below the column, closer to the second row
        pos = axs[1, c].get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.04, pos.width, 0.013])
        cbar = fig.colorbar(contour_orig, cax=cbar_ax, orientation='horizontal')
        if var == 'prmsl':
            cbar.set_ticks(levels[var][::2])  # Show fewer ticks for clarity, all integers
        cbar.set_label(var)

    # Save the figure
    output_image = f'MESO_predict_frame_{t:02d}.png'
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Saved {output_image}")
    plt.close(fig)