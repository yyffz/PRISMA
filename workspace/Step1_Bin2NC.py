import glob
import datetime
import numpy as np
import argparse
import xarray as xr
import pandas as pd
import os
import bz2
import struct
from datetime import datetime, timedelta
from dask.diagnostics import ProgressBar


# 分块处理函数
def chunk_time(ds):
    # 创建一个新的字典dims，包含xarray数据集ds的所有维度。
    dims = {k: v for k, v in ds.sizes.items()}
    dims['time'] = 1  # 将时间维度的块大小设置为1
    ds = ds.chunk(dims)  # 对数据集进行分块处理，以优化计算性能。
    return ds


# 获取日期
def get_date():
    # 根据命令行参数start_str和end_str指定的开始和结束时间，生成一个日期列表。
    start_time = datetime.datetime.strptime(args.start_str, "%Y%m%d%H%M%S")
    end_time = datetime.datetime.strptime(args.end_str, "%Y%m%d%H%M%S")
    dates = []
    # 使用pandas.date_range生成一个日期范围，然后格式化每个日期并将其添加到列表中。
    for date in pd.date_range(start_time, end_time, freq="15min"):
        date = date.strftime("%Y%m%d%H%M%S")
        dates.append(date)
    return dates


# 解析雷达数据
class MosaicParser(object):
    def __init__(self, filename, clip=False, wlon2=73, elon2=135, slat2=12.2, nlat2=54.2, ):
        super(MosaicParser, self).__init__()
        self.varname = None  # 存储文件中数据的变量名称
        self.pictime = None  # 存储文件中数据的图片时间
        self.clip = clip
        self.wlon2, self.elon2, self.slat2, self.nlat2 = wlon2, elon2, slat2, nlat2
        self.filename = filename

    # 静态方法parse用于解析字节缓冲区buff中的数据
    @staticmethod
    def parse(buff):
        radar_data_list, head_data = None, {}
        if True:
            compress_flag = struct.unpack('h', buff[166:168])[0]  # 字节位置166-167,数据压缩标识, 0=无,1=bz2,2=zip,3=lzw
            cols = struct.unpack('i', buff[148:152])[0]  # 格点坐标为列数
            rows = struct.unpack('i', buff[152:156])[0]  # 格点坐标为行数

            pdt_code = struct.unpack('h', buff[12:14])[0]  # 12-13 拼图产品编号
            pdt_bytes = struct.unpack('i', buff[92:96])
            # print(f'compress_flag:{compress_flag}, nrows:{rows}, cols:{cols}, pdt_bytes:{pdt_bytes}')
            # print(compress_flag, rows, cols, struct.unpack('i', buff[92:96]))  # 产品数据字节数
            compress_data = buff[256:]  # 文件头共占用 256 字节，从256开始是压缩的数据

        if compress_flag == 1:
            data = bz2.decompress(compress_data)
            if not data:
                return None
        else:
            data = compress_data

        data_array = np.frombuffer(data, dtype=np.int16)  # frombuffer将data以流的形式读入转化成ndarray对象
        data_array = np.reshape(data_array, (rows, cols))  # numpy.reshape(a, newshape, order=’C’)

        l_lat = struct.unpack('i', buff[124:128])[0]  # 数据南边界，单位：1/1000度，放大1千倍
        l_lon = struct.unpack('i', buff[128:132])[0]  # 数据西边界
        u_lat = struct.unpack('i', buff[132:136])[0]  # 数据北边界
        u_lon = struct.unpack('i', buff[136:140])[0]  # 数据东边界
        height = struct.unpack('h', buff[164:166])[0]  # 雷达高度
        head_data['pdt_code'] = pdt_code
        return data_array, l_lat / 1000., l_lon / 1000., u_lat / 1000., u_lon / 1000., head_data, height

    # 获取数据
    def getdata(self):
        pictime: object = self.filename.split('/')[-1].split('_')[9] + \
                          self.filename.split('/')[-1].split('_')[10].split('.')[0]
        varname = self.filename.split('/')[-1].split('_')[8]

        f = open(self.filename, 'rb')
        buf = f.read()
        f.close()
        data_arr, llat, llon, ulat, ulon, head, height = self.parse(buf)
        if self.clip:
            data_arr = data_arr[::-1]  # 将数据逆序
            begin_col = int((self.wlon2 - llon) / 0.01)
            end_col = int((self.elon2 - llon) / 0.01)
            begin_row = int((self.slat2 - llat) / 0.01)
            end_row = int((self.nlat2 - llat) / 0.01)
            region_data = data_arr[begin_row: end_row, begin_col: end_col]

            region_data1 = region_data.copy()
            region_data1 = region_data1.astype(float)
            region_data1[region_data1 < 0.01] = 0
            lon_sin = np.linspace(self.wlon2, self.elon2, region_data1.shape[1])
            lat_sin = np.linspace(self.slat2, self.nlat2, region_data1.shape[0])
            lons_arr, lats_arr = np.meshgrid(lon_sin, lat_sin) 
            region_data = region_data1

        else:
            data_arr = data_arr[::-1]
            region_data = data_arr.copy()
            # 保留大于0.01的值，小于0.01的值赋值为0
            region_data = region_data.astype(float)
            region_data[region_data < 0.01] = 0

            lon_sin = np.linspace(llon, ulon, region_data.shape[1])
            lat_sin = np.linspace(llat, ulat, region_data.shape[0])
            lons_arr, lats_arr = np.meshgrid(lon_sin, lat_sin)
        return lons_arr, lats_arr, region_data / 10


def create_nc(arr_lons, arr_lats, var_arr, time, filename):
    lon = arr_lons[0, :].astype(np.float32)  # 假设所有行的经度值相同，取第一行
    lat = arr_lats[:, 0].astype(np.float32)  # 假设所有列的纬度值相同，取第一列
    # print(f'lon.shape():{lon.shape}, lat.shape:{lat.shape}')
    # print(f'var_arr.shape:{var_arr.shape}')

    ds = xr.Dataset(
        {
            "CREF": (["time", "lat", "lon"], var_arr[np.newaxis, :, :].astype(np.float32)),
        },
        coords={
            "time": np.array([time], dtype='datetime64[ns]'),
            "lat": lat,
            "lon": lon,
        }
    )

    ds = chunk_time(ds)
    ds['CREF'].encoding.update({'dtype': 'float32'})
    # 打印ds['CREF']的数值范围
    print(f"ds['CREF'].max(): {ds['CREF'].max().values}, ds['CREF'].min(): {ds['CREF'].min().values}, ds['CREF'].mean(): {ds['CREF'].mean().values}")
    
    delayed_ds = ds.to_netcdf(filename, engine="netcdf4", compute=False)
    with ProgressBar():  # 上下文管理器，用于在执行长时间运行的任务时显示进度条。
        delayed_ds.compute()  # 显式触发计算
    print(f'save2nc_path:{filename}')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="datain dir", default='/public/DataWarehouse/OBS/CMAMESO/radarl3crefqc/2024/08/08/')
    parser.add_argument('--save_dir', type=str, help="dataout dir", default="./")
    parser.add_argument('--start_str', type=str, help="%Y%m%d%H%M%S", default='20220801000000') 
    parser.add_argument('--end_str', type=str, help="%Y%m%d%H%M%S", default='20220830000000') 
    parser.add_argument('--start_lon', type=float, default='114') 
    parser.add_argument('--start_lat', type=float, default='28') 
    parser.add_argument('--rect', type=float, nargs="+", default=[])
    args = parser.parse_args()

    NowTime = datetime.strptime(args.start_str, "%Y%m%d%H%M%S")
    EndTime = datetime.strptime(args.end_str, "%Y%m%d%H%M%S")
    while NowTime < EndTime:
        print('NowTime: ', NowTime)
        yrmn = NowTime.strftime("%Y%m")
        # date_str = NowTime.strftime("%Y%m%d_%H%M%S")
        date_str = NowTime.strftime("%Y%m%d_%H%M")
        save_path = f'{args.save_dir}/CREF_with_time_{date_str}.nc'
        # print(f'save_path: {save_path}')

        # if os.path.exists(save_path):
        #     print(f'File {save_path} already exists, skipping to next time step.')
        #     NowTime = NowTime + timedelta(minutes=6)
        #     continue

        filename = glob.glob(f'{args.data_dir}/Z_RADA_*_CREF_{date_str}*.bin')
        print(f'{args.data_dir}/Z_RADA_*_CREF_{date_str}*.bin')
        # print(filename[0])
        if len(filename) == 0:
            print(f'No file found for {date_str}, skipping to next time step.')
            NowTime = NowTime + timedelta(minutes=6)
            continue

        # Mosaic = MosaicParser(filename[0], clip=True, wlon2=args.start_lon, elon2=args.start_lon+10.245, 
        #                       slat2=args.start_lat, nlat2=args.start_lat+6.4) # (1000, 1000)
        Mosaic = MosaicParser(filename[0], clip=False)
        arr_lons, arr_lats, region_arr = Mosaic.getdata()
        print(f'arr_lons.shape:{arr_lons.shape}')  # (纬度, 经度)
        # 打印region_arr的数值范围
        print(f'region_arr.shape:{region_arr.shape}')
        print(f'region_arr.max():{region_arr.max()}, region_arr.min():{region_arr.min()}, region_arr.mean():{region_arr.mean()}')
        
        # print("########################################")

        create_nc(arr_lons, arr_lats, region_arr, NowTime, save_path)
        # break

        NowTime = NowTime + timedelta(minutes=6)


