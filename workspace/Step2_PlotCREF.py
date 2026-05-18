import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_distribution_CREF(lon_contourf, lat_contourf, data_contourf, pictime):
    fig = plt.figure(figsize=(8, 6.5), dpi=400)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    bwith = 1.5
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)

    # WestLon, EastLon, SouthLat, NorthLat = [104, 108, 26, 28]
    # WestLon, EastLon, SouthLat, NorthLat = [100, 110.24, 24, 30.4]
    # ax.set_xticks(np.arange(WestLon, EastLon + 1, 1), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(SouthLat, NorthLat + 1, 0.5), crs=ccrs.PlateCarree())

    WestLon, EastLon, SouthLat, NorthLat = [111, 121.2, 36, 42.4]
    Extent = [WestLon, EastLon, SouthLat, NorthLat]
    ax.set_extent(Extent, crs=ccrs.PlateCarree())

    ax.set_xticks(np.arange(111, 121.2, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(36, 42.4, 2), crs=ccrs.PlateCarree())

    ax_position = ax.get_position()
    title_y = ax_position.y1 + 0.1  # 在子图上方一点的位置
    dt_pictime = pd.to_datetime(pictime, utc=True)
    formatted_pictime = dt_pictime.strftime('%Y-%m-%d %H:%M:%S')
    plt.suptitle(f'CREF: {formatted_pictime}', fontsize=14, y=title_y)

    left = ax_position.x1 + 0.01  # 增加从图的右边缘到colorbar左边缘的距离
    bottom = ax_position.y0
    width = 0.02  # colorbar宽度，可以根据需要调整
    height = ax_position.height



    levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]  # 数值多一个
    colors = ["#5a9bea", "#94f75f", "#62d53f", "#3f8e27", "#ffff54", "#FFE400", "#e1c140",
              "#FBAA41", "#F98921", "#FD6841", "#ea3323", "#c4291c", "#b02418", "#ea33e8", "#891aae", "#a891ea"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cs = ax.contourf(lon_contourf, lat_contourf, data_contourf,
                     levels=levels, cmap=cmap, norm=norm, extend='max',
                     transform=ccrs.PlateCarree())  # 'extend' 控制如何显示超出 levels 范围的区域

    ax.gridlines(linewidth=0.6, color='black', alpha=0.5, linestyle='--')
    # shp_file_path = os.path.join('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/plot/shp/china2.dbf')
    shp_file_path = os.path.join('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/plot_use/dijishi_2004.dbf')
    shp = shpreader.Reader(shp_file_path).geometries()
    ax.add_geometries(shp, ccrs.PlateCarree(), facecolor='None', edgecolor='black', zorder=1, lw=0.4, alpha=0.9)
    shp_file_path = os.path.join('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/plot_use/river1.dbf')
    shp = shpreader.Reader(shp_file_path).geometries()
    ax.add_geometries(shp, ccrs.PlateCarree(), facecolor='None', edgecolor='b', zorder=1, lw=0.4, alpha=0.9)

    # 添加垂直bar
    cbar_ax = fig.add_axes([left, bottom, width, height])
    cbar = plt.colorbar(cs, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    
    ax.plot(106.09, 26.81,  # 经纬度坐标（注意顺序：经度在前，纬度在后）
        marker='^',      # 星形标记
        color='red',     # 红色
        markersize=8,    # 标记大小
        alpha=1,       # 透明度
        transform=ccrs.PlateCarree(), # 指定为地理坐标系
        zorder=10)       # 确保标记在最上层显示
    
    # plt.show()
    plt.savefig(f"./CREF_{dt_pictime.strftime('%Y%m%d_%H%M')}.png")
    return None


def PlotAxis(ax):
    bwith = 1
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)

    # WestLon, EastLon, SouthLat, NorthLat = [114, 124.2, 28, 34.4]
    # Extent = [WestLon, EastLon, SouthLat, NorthLat]
    # ax.set_extent(Extent, crs=ccrs.PlateCarree())
    # ax.set_xticks(np.arange(114, 124.2, 2), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(28, 34.4, 2), crs=ccrs.PlateCarree())
    
    # WestLon, EastLon, SouthLat, NorthLat = [70, 140.1, 16, 54.1]
    # Extent = [WestLon, EastLon, SouthLat, NorthLat]
    # ax.set_extent(Extent, crs=ccrs.PlateCarree())
    # ax.set_xticks(np.arange(70, 140.1, 10), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(15, 54.1, 5), crs=ccrs.PlateCarree())
    
    WestLon, EastLon = 113.0, 123.24
    SouthLat, NorthLat = 27.0, 33.4
    Extent = [WestLon, EastLon, SouthLat, NorthLat]
    ax.set_extent(Extent, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(113, 124, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(27, 34, 1), crs=ccrs.PlateCarree())
    
    # WestLon, EastLon, SouthLat, NorthLat = [100, 127.1, 22, 35.1]
    # Extent = [WestLon, EastLon, SouthLat, NorthLat]
    # ax.set_extent(Extent, crs=ccrs.PlateCarree())
    # ax.set_xticks(np.arange(100, 127.1, 5), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(22, 35.1, 2), crs=ccrs.PlateCarree())
    
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=7)
    return None


def plot_CREF(CREF_value, CREF_lat, CREF_lon, fig_name):
    plt.rcParams['font.sans-serif']=['Times New Roman']
    fig = plt.figure(figsize=(7.5, 5), dpi=400)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    PlotAxis(ax)

    # 定义雷达量级和对应颜色
    levels = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    colors = ["#94f75f", "#62d53f", "#3f8e27", "#ffff54", "#FFE400", "#e1c140",
              "#FBAA41", "#F98921", "#FD6841", "#ea3323", "#c4291c", "#b02418", "#ea33e8", "#891aae", "#a891ea"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    contour = ax.contourf(CREF_lon, CREF_lat, CREF_value,
                          levels=levels, cmap=cmap, norm=norm, extend='max',
                          transform=ccrs.PlateCarree())
    cbar = plt.colorbar(contour, orientation='vertical', pad=0.03, aspect=35, shrink=0.6)
    cbar.set_label('dBZ')

    ax.gridlines(linewidth=0.6, color='black', alpha=0.9, linestyle='--')
    # shp_file_path = os.path.join('/public/home/sunhaofei/plot_use/dijishi_2004.dbf')
    shp_file_path = os.path.join('/public/home/sunhaofei/plot_use/china1.shp')
    shp = shpreader.Reader(shp_file_path).geometries()
    ax.add_geometries(shp, ccrs.PlateCarree(), facecolor='None', edgecolor='black', zorder=1, lw=0.6, alpha=0.9)
    
    # Parse UTC time and compute Beijing time (UTC+8)
    utc_time = datetime.fromisoformat(fig_name)
    beijing_time = utc_time + timedelta(hours=8)
    utc_str = utc_time.strftime('%Y-%m-%d %H:%M UTC')
    beijing_str = beijing_time.strftime('%Y-%m-%d %H:%M BJT')
    plt.title(f'{utc_str}\n{beijing_str}', fontsize=6)
    plt.xlabel('Longitude', fontsize=6)
    plt.ylabel('Latitude', fontsize=6)
    plt.savefig(f'./{fig_name}.png', dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close()


# ds = xr.open_mfdataset("../datasets/Radar_China/202403/CREF_with_time_*.nc")
# ds = xr.open_mfdataset("../datasets/Radar_SH/202409/CREF_with_time_20240919_1*.nc")
# ds = xr.open_mfdataset("../datasets/Radar_SH/202304/CREF_with_time_20230415_0*.nc")
ds = xr.open_mfdataset("../case_0825/CREF_with_time_*.nc")
print(ds)
print(len(ds.time))

# for time in ds.time[:50]:
for time in ds.time[:]:
    ds_t = ds.sel(time=time)
    # print(ds_t)
    # print(ds_t.CREF.shape)
    time_value = str(time.values)[:19]
    print(f"Processing time: {time_value}")
    plot_CREF(ds_t.CREF, ds_t.lat, ds_t.lon, fig_name=time_value)



