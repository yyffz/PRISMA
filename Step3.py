import xarray as xr
import numpy as np
import pandas as pd
import torch
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.mpl.ticker as cticker
from matplotlib.colors import BoundaryNorm
from datetime import datetime


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
    WestLon, EastLon, SouthLat, NorthLat = [108.64, 124, 22, 37.36]
    Extent = [WestLon, EastLon, SouthLat, NorthLat]
    ax.set_extent(Extent, crs=ccrs.PlateCarree())

    ax.set_xticks(np.arange(108, 125, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(22, 38, 2), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=7)
    return None


# Load and subset the NetCDF file
file_path = '/public/home/sunhaofei/cosmos-predict1/workspace/Tokenizer_prep/MESO_20230428.nc'
ds = xr.open_dataset(file_path)
print(ds)

# Subset the dataset
ds = ds.sel(time=slice('2023-04-28T00:00', '2023-04-28T04:00'), lat=slice(37.36, 22), lon=slice(108.64, 124))
print(ds)

# Select variables and convert to tensor
vars_selected = ['prmsl', '10u', '10v']
ds_stacked = ds[vars_selected].to_array(dim='channel')  # Shape: (channel:3, time:5, lat:512, lon:512)
data_np = ds_stacked.values  # Shape: (3, 5, 512, 512)

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

# Convert to torch tensor
input_tensor = torch.from_numpy(data_np).unsqueeze(0).to('cuda').to(torch.bfloat16)  # Shape: (1, 3, 5, 512, 512), [0..1]

# Encode and decode
model_name = "Cosmos-Tokenize1-CV4x8x8-360p"
input_tensor = input_tensor * 2. - 1.  # Normalize to [-1..1]
encoder = CausalVideoTokenizer(checkpoint_enc=f'checkpoints/{model_name}/encoder.jit')  # [1, 16, 2, 64, 64]
(latent,) = encoder.encode(input_tensor)
decoder = CausalVideoTokenizer(checkpoint_dec=f'checkpoints/{model_name}/decoder.jit')
reconstructed_tensor = decoder.decode(latent)  # [-1..1]

# Scale back to original data range
reconstructed_tensor = reconstructed_tensor.to('cpu').to(torch.float32)
normalized = (reconstructed_tensor + 1) / 2  # Reverse [-1,1] to [0,1]
normalized = normalized.squeeze(0)  # Shape: (3, 5, 512, 512)
reconstructed_np = normalized.numpy()
for c in range(reconstructed_np.shape[0]):
    reconstructed_np[c] = reconstructed_np[c] * (maxs[c] - mins[c]) + mins[c]

# Save reconstructed tensor
output_path = f'./reconstructed_MESO_{model_name}.npy'
np.save(output_path, reconstructed_np)
print(f"Reconstructed tensor saved to {output_path}")

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
for t in range(5):
    # Set up figure for this time step
    fig, axs = plt.subplots(2, 3, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.tight_layout(pad=5.0, h_pad=8.0, rect=[0, 0.1, 1, 0.95])  # Larger row gap, reserve space at bottom

    time_str = pd.Timestamp(ds.time.values[t]).strftime('%Y-%m-%d %H:%M')
    print(f"Processing time: {time_str}")
    
    for c, var in enumerate(vars_selected):
        print(f"Processing variable: {var}, channel: {c}, time step: {t}")
        print(f"Original data range for {var}: min={mins[c]}, max={maxs[c]}")
        print(f"Reconstructed data range for {var}: min={np.nanmin(reconstructed_np[c, t])}, max={np.nanmax(reconstructed_np[c, t])}")
        
        # Prepare data for plotting (handle units for prmsl)
        original_data = ds[var].isel(time=t)
        recon_data = reconstructed_np[c, t]
        if var == 'prmsl' and maxs[c] > 10000:  # Convert to hPa for plotting
            original_data = original_data / 100
            recon_data = recon_data / 100

        # Plot original data
        PlotAxis(axs[0, c])
        norm = BoundaryNorm(levels[var], ncolors=plt.cm.jet.N, clip=True)
        contour_orig = axs[0, c].contourf(ds.lon, ds.lat, original_data, levels=levels[var], 
                                         cmap='jet', norm=norm, extend='both', transform=ccrs.PlateCarree())
        axs[0, c].gridlines(linewidth=0.6, color='black', alpha=0.5, linestyle='--')
        shp_file_path = '/public/home/sunhaofei/plot_use/china1.shp'
        shp = shpreader.Reader(shp_file_path).geometries()
        axs[0, c].add_geometries(shp, ccrs.PlateCarree(), facecolor='None', edgecolor='black', zorder=1, lw=0.6, alpha=0.4)
        axs[0, c].set_title(f'Original {var} {time_str} UTC', fontsize=9)

        # Plot reconstructed data
        PlotAxis(axs[1, c])
        contour_recon = axs[1, c].contourf(ds.lon, ds.lat, recon_data, levels=levels[var], 
                                          cmap='jet', norm=norm, extend='both', transform=ccrs.PlateCarree())
        axs[1, c].gridlines(linewidth=0.6, color='black', alpha=0.5, linestyle='--')
        shp_file_path = '/public/home/sunhaofei/plot_use/china1.shp'
        shp = shpreader.Reader(shp_file_path).geometries()
        axs[1, c].add_geometries(shp, ccrs.PlateCarree(), facecolor='None', edgecolor='black', zorder=1, lw=0.6, alpha=0.4)
        axs[1, c].set_title(f'Reconstructed {var} {time_str} UTC', fontsize=9)

        # Shared colorbar below the column, closer to the second row
        # cbar_ax = fig.add_axes([axs[1, c].get_position().x0, 0.05, axs[1, c].get_position().width, 0.03])
        pos = axs[1, c].get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.04, pos.width, 0.013])
        cbar = fig.colorbar(contour_orig, cax=cbar_ax, orientation='horizontal')
        if var == 'prmsl':
            cbar.set_ticks(levels[var][::2])  # Show fewer ticks for clarity, all integers
        cbar.set_label(var)

    # Save the figure
    output_image = f'frame_{t:02d}.png'
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Saved {output_image}")
    plt.close(fig)