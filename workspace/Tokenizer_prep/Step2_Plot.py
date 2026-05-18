import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# Define the plotting function for 2t
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

    # Set extent based on provided lat/lon ranges
    WestLon, EastLon, SouthLat, NorthLat = [70, 145, 10, 60.1]
    Extent = [WestLon, EastLon, SouthLat, NorthLat]
    ax.set_extent(Extent, crs=ccrs.PlateCarree())

    ax.set_xticks(np.arange(70, 146, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(10, 61, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=7)
    return None


def plot_2t(temp_value, temp_lat, temp_lon, fig_name):
    fig = plt.figure(figsize=(10, 8), dpi=400)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    PlotAxis(ax)

    # Define temperature levels and colors (in Celsius)
    levels = np.arange(-10, 40, 2)
    cmap = plt.cm.jet
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Plot temperature contour
    contour = ax.contourf(temp_lon, temp_lat, temp_value, levels=levels, 
                         cmap=cmap, norm=norm, extend='both', 
                         transform=ccrs.PlateCarree())
    cbar = plt.colorbar(contour, orientation='vertical', pad=0.03, aspect=35, shrink=0.6)
    cbar.set_label('Temperature (°C)')

    # Add gridlines and shapefile
    ax.gridlines(linewidth=0.6, color='black', alpha=0.5, linestyle='--')
    shp_file_path = '/public/home/sunhaofei/plot_use/china1.shp'
    shp = shpreader.Reader(shp_file_path).geometries()
    ax.add_geometries(shp, ccrs.PlateCarree(), facecolor='None', edgecolor='black', zorder=1, lw=0.6, alpha=0.4)

    plt.title(f'2m Temperature {fig_name} UTC', fontsize=9)
    plt.xlabel('Longitude', fontsize=9)
    plt.ylabel('Latitude', fontsize=9)
    plt.savefig(f'./{fig_name}_2t.png', dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close()

# Load GRIB file and extract variables
file_path = '/public/home/sunhaofei/cosmos-predict1/workspace/Tokenizer_prep/MESO_20230428.nc'
ds = xr.open_dataset(file_path)
print(ds)

lat_range = slice(37.36, 22)  # 注意反转：从高纬到低纬，以匹配降序坐标
lon_range = slice(108.64, 124)  
ds = ds.sel(time=slice('2023-04-28T00:00', '2023-04-28T04:00'), lat=lat_range, lon=lon_range)

# plot_2t(ds['2t'], ds.lat, ds.lon, '202304280000')

import torch
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

vars_selected = ['prmsl', '10u', '10v']
ds_stacked = ds[vars_selected].to_array(dim='channel')  # Shape: (channel:3, time:5, lat:512, lon:512)
data_np = ds_stacked.values  # Shape: (3, 5, 512, 512)
# For each channel, normalize to [0,1] based on min/max across all times
for c in range(data_np.shape[0]):
    min_val = np.min(data_np[c])
    max_val = np.max(data_np[c])
    if max_val > min_val:
        data_np[c] = (data_np[c] - min_val) / (max_val - min_val)

input_tensor = torch.from_numpy(data_np).unsqueeze(0).to('cuda').to(torch.bfloat16)  # Shape: (1, 3, 5, 512, 512)
print(f"Input tensor shape: {input_tensor.shape}")
print(f"Input tensor max: {input_tensor.max().item()}, min: {input_tensor.min().item()}, mean: {input_tensor.mean().item()}")
model_name = "Cosmos-Tokenize1-CV4x8x8-360p"
# input_tensor = torch.rand(1, 3, 5, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]


input_tensor = input_tensor * 2. - 1.  # Normalize to [-1..1]
encoder = CausalVideoTokenizer(checkpoint_enc=f'checkpoints/{model_name}/encoder.jit')
(latent,) = encoder.encode(input_tensor)
torch.testing.assert_close(latent.shape, (1, 16, 3, 64, 64))

# The input tensor can be reconstructed by the decoder as:
decoder = CausalVideoTokenizer(checkpoint_dec=f'checkpoints/{model_name}/decoder.jit')
reconstructed_tensor = decoder.decode(latent)
torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
print(f"Reconstructed tensor max: {reconstructed_tensor.max().item()}, min: {reconstructed_tensor.min().item()}, mean: {reconstructed_tensor.mean().item()}") 


# # save the reconstructed tensor
# output_path = f'./reconstructed_{model_name}.npy'
# np.save(output_path, reconstructed_tensor.cpu().numpy())
# print(f"Reconstructed tensor saved to {output_path}")