import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def check_data_info(data_path):
    """检查数据的shape和数值范围"""
    data = np.load(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data min: {np.min(data)}")
    print(f"Data max: {np.max(data)}")
    print(f"Data mean: {np.mean(data)}")
    print(f"Data std: {np.std(data)}")
    return data


def denormalize_data(data, min_val=0.0, max_val=70.0):
    """将标准化的数据反标准化到指定范围
    
    Args:
        data: 标准化后的数据 (通常在0-1范围内)
        min_val: 目标最小值 (dBZ)
        max_val: 目标最大值 (dBZ)
    
    Returns:
        反标准化后的数据
    """
    # 将数据从[0,1]范围映射到[min_val, max_val]范围
    denormalized = data * (max_val - min_val) + min_val
    
    # 确保数据在合理范围内
    denormalized = np.clip(denormalized, min_val, max_val)
    
    print(f"After denormalization - min: {np.min(denormalized):.2f}, max: {np.max(denormalized):.2f}, mean: {np.mean(denormalized):.2f}")
    
    return denormalized


def create_coordinate_grids(lat_range, lon_range, data_shape):
    """根据数据shape和空间范围创建经纬度网格"""
    # 假设数据的最后两个维度是空间维度 (lat, lon)
    nlat, nlon = data_shape[-2], data_shape[-1]
    
    # 创建经纬度数组
    lat = np.linspace(lat_range[0], lat_range[1], nlat)
    lon = np.linspace(lon_range[0], lon_range[1], nlon)
    
    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    return lat_grid, lon_grid, lat, lon


def plot_cref_prediction(data_2d, lat_grid, lon_grid, time_idx, base_time, output_dir):
    """绘制单个时刻的CREF预测结果"""
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    fig = plt.figure(figsize=(8, 6.5), dpi=400)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # 设置边框
    bwith = 1.5
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    
    # 设置地图范围 - 使用指定的空间范围
    WestLon, EastLon = 113.0, 123.24
    SouthLat, NorthLat = 27.0, 33.4
    Extent = [WestLon, EastLon, SouthLat, NorthLat]
    ax.set_extent(Extent, crs=ccrs.PlateCarree())
    
    # 设置刻度
    ax.set_xticks(np.arange(113, 124, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(27, 34, 1), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 定义雷达量级和对应颜色
    levels = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    colors = ["#94f75f", "#62d53f", "#3f8e27", "#ffff54", "#FFE400", "#e1c140",
              "#FBAA41", "#F98921", "#FD6841", "#ea3323", "#c4291c", "#b02418", 
              "#ea33e8", "#891aae", "#a891ea"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    # 绘制等值线填充图
    contour = ax.contourf(lon_grid, lat_grid, data_2d,
                          levels=levels, cmap=cmap, norm=norm, extend='max',
                          transform=ccrs.PlateCarree())
    
    # 添加colorbar
    cbar = plt.colorbar(contour, orientation='vertical', pad=0.03, aspect=35, shrink=0.6)
    cbar.set_label('dBZ', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 添加网格线
    ax.gridlines(linewidth=0.6, color='black', alpha=0.5, linestyle='--')
    
    # 添加地理边界（如果shapefile存在）
    try:
        shp_file_path = '/public/home/sunhaofei/plot_use/china1.shp'
        if os.path.exists(shp_file_path):
            shp = shpreader.Reader(shp_file_path).geometries()
            ax.add_geometries(shp, ccrs.PlateCarree(), facecolor='None', 
                            edgecolor='black', zorder=1, lw=0.6, alpha=0.9)
    except:
        print("Warning: Could not load shapefile for geographic boundaries")
    
    # 计算时间（每个时刻间隔6分钟）
    prediction_time = base_time + timedelta(minutes=6 * time_idx)
    utc_str = prediction_time.strftime('%Y-%m-%d %H:%M UTC')
    beijing_time = prediction_time + timedelta(hours=8)
    beijing_str = beijing_time.strftime('%Y-%m-%d %H:%M BJT')
    
    plt.title(f'CREF Prediction (T+{time_idx*6:03d}min)\n{utc_str}\n{beijing_str}', fontsize=12)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    
    # 保存图片 - 使用起报时间作为基准
    output_filename = f'CREF_pred_{base_time.strftime("%Y%m%d_%H%M")}_T+{time_idx*6:03d}min.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"Saved: {output_filename}")


def main():
    # 数据路径
    data_path = "/public/home/sunhaofei/cosmos-predict1/AR_pred_08250600.npy"
    output_dir = "/public/home/sunhaofei/cosmos-predict1/workspace/AR_prediction_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 空间范围（从Step4_SelectCase.py获取）
    lat_range = (27.0, 33.4)
    lon_range = (113.0, 123.24)
    
    # 基准时间（2025年8月25日06:00 UTC）
    base_time = datetime(2025, 8, 25, 6, 0, 0)
    print(f"Base time: {base_time.strftime('%Y-%m-%d %H:%M UTC')}")
    
    print("Loading and checking data...")
    # 检查数据信息
    data = check_data_info(data_path)
    
    print(f"\nData loaded successfully!")
    print(f"Expected to process {data.shape[0] if len(data.shape) > 2 else 1} time steps")
    
    # 根据数据维度处理
    if len(data.shape) == 4:  # (batch, time, lat, lon)
        data = data[0]  # 取第一个batch
        num_times = data.shape[0]
    elif len(data.shape) == 3:  # (time, lat, lon)
        num_times = data.shape[0]
    else:
        print(f"Unexpected data shape: {data.shape}")
        return
    
    # 反标准化数据到dBZ范围 (0-70)
    print("\nDenormalizing data to dBZ range (0-70)...")
    data = denormalize_data(data, min_val=0.0, max_val=70.0)
    print(f"\nProcessing {num_times} time steps...")
    
    # 创建坐标网格
    lat_grid, lon_grid, lat, lon = create_coordinate_grids(lat_range, lon_range, data.shape)
    
    print(f"Coordinate grids created:")
    print(f"Latitude range: {lat[0]:.2f} to {lat[-1]:.2f}")
    print(f"Longitude range: {lon[0]:.2f} to {lon[-1]:.2f}")
    print(f"Grid shape: {lat_grid.shape}")
    
    # 绘制每个时刻
    for t in range(num_times):
        data_2d = data[t]
        print(f"\nProcessing time step {t+1}/{num_times}")
        print(f"Time step data - min: {np.min(data_2d):.2f}, max: {np.max(data_2d):.2f}, mean: {np.mean(data_2d):.2f}")
        
        plot_cref_prediction(data_2d, lat_grid, lon_grid, t, base_time, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Total {num_times} images generated.")


if __name__ == "__main__":
    main()