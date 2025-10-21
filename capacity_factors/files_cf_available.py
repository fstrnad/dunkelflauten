# %%
import pandas as pd
import numpy as np
import xarray as xr
import os
import geoutils.preprocessing.open_nc_file as of
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
from importlib import reload
import yaml

# %%
if os.getenv("HOME") == '/home/ludwig/fstrnad80':
    cmip6_dir = "/mnt/lustre/work/ludwig/shared_datasets/CMIP6/"
    data_dir = f'{cmip6_dir}/downscaling/'
    era5_dir = "/mnt/lustre/work/ludwig/shared_datasets/climate_data/Europe"
    with open('./config_cluster.yaml', 'r') as file:
        config = yaml.safe_load(file)
else:
    plot_dir = "/home/strnad/plots/dunkelflauten/downscaling_cmip6/"
    data_dir = "/home/strnad/data/CMIP6/downscaling/"
    cmip6_dir = "/home/strnad/data/CMIP6/"
    era5_dir = "/home/strnad/data/climate_data/Europe"
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
# %%
reload(of)
reload(gut)

country_name = 'Germany'
gs_dws = 1.0
fine_res = 0.25
coarse_res = 1.0
N = 3
gcm = 'GFDL-ESM4'
gcm = 'MPI-ESM1-2-HR'
tr_historical = [
    ('1980-01-01', '1990-01-01'),
    ('1990-01-01', '2000-01-01'),
    ('2000-01-01', '2010-01-01'),
    ('2010-01-01', '2015-01-01')]

tr_ssp = [
    ('2020-01-01', '2030-01-01'),
    ('2030-01-01', '2040-01-01'),
    # ('2040-01-01', '2050-01-01'),
    ('2050-01-01', '2060-01-01'),
    # ('2060-01-01', '2070-01-01'),
    ('2070-01-01', '2080-01-01'),
    # ('2080-01-01', '2090-01-01'),
    ('2090-01-01', '2100-01-01')
]
ssps = ['historical', 'ssp245', 'ssp585']
gcms = [
    'MPI-ESM1-2-HR',
    'GFDL-ESM4',
    'MIROC6',
    'IPSL-CM6A-LR',
    'CanESM5'
]

for gcm in gcms:
    for ssp in ssps:
        time_ranges = tr_historical if ssp == 'historical' else tr_ssp
        gcm_str = f'{gcm}_{ssp}'
        for (start_date, end_date) in time_ranges:
            tr_str = f'{start_date}_{end_date}'

            sample_folder = f'{data_dir}/{gs_dws}/{gcm_str}/{tr_str}_N{N}/'
            sample_path_fine = fut.get_files_in_folder(sample_folder,
                                                       include_string=f'{fine_res}')
            sample_path_coarse = fut.get_files_in_folder(
                sample_folder,
                include_string=f'{coarse_res}')

            for savepath_samples in [sample_path_fine,
                                     sample_path_coarse]:
                if len(savepath_samples) == 0:
                    print(f'Samples {gcm} {ssp} {tr_str} does not exist')

# %%
