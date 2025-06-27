# %%
import cf_utils as cfu
import geoutils.utils.met_utils as mut
import geoutils.geodata.solar_radiation as sr
import workaround_fsr as wf
import geoutils.utils.statistic_utils as sut
from scipy import stats
import pandas as pd
import numpy as np
import xarray as xr
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import atlite as at
from importlib import reload
import os

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
reload(cfu)

country_name = 'Germany'
gs_dws = 1.0
gs = 0.25
fine_res = 0.25
gs_era5 = 0.25  # resolution of the ERA5 data
coarse_res = 1.0
N = 3

tr_historical = [('1980-01-01', '2015-01-01')]  # full range
tr_ssp = [('2020-01-01', '2100-01-01')]  # full range

ssps = [
    'historical',
    'ssp245',
    'ssp585'
]

gcms = [
    'MPI-ESM1-2-HR',
    'GFDL-ESM4',
    'MIROC6',
    'IPSL-CM6A-LR',
    'CanESM5'
]

overwrite = False  # Set to True if you want to overwrite existing files
use_bc = True  # Set to True if you want to use bias-corrected data
# %%
reload(cfu)
for gcm in gcms:
    for ssp in ssps:
        time_ranges = tr_historical if ssp == 'historical' else tr_ssp
        gcm_str = f'{gcm}_{ssp}'
        start_date, end_date = time_ranges[0]
        tr_str = f'{start_date}_{end_date}'

        gcm_str = f'{gcm}_{ssp}'
        folder_cmip6 = f'{data_dir}/{gs_dws}/{gcm_str}/{tr_str}_N{N}/'
        cmip6_file_fine = f'samples_{gcm_str}_{tr_str}_{gs_era5}_log_False'
        if use_bc:
            cmip6_file_fine += '_bc'
        filename_cmip6 = f'{folder_cmip6}/{cmip6_file_fine}.nc'
        sample_path_coarse_files = fut.get_files_in_folder(
            folder_cmip6,
            include_string=f'{coarse_res}')
        if len(sample_path_coarse_files) > 0:
            sample_path_coarse = sample_path_coarse_files[0]
        else:
            continue

        if use_bc:
            savepath_dict_fine = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{fine_res}_{tr_str}_bc.npy'
        else:
            savepath_dict_fine = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{fine_res}_{tr_str}.npy'
        savepath_dict_coarse = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{coarse_res}_{tr_str}.npy'

        for savepath_dict in [savepath_dict_fine,
                              savepath_dict_coarse]:
            if not fut.exist_file(savepath_dict) or overwrite:
                ds_cmip_raw_fine = of.open_nc_file(
                    filename_cmip6, verbose=False)
                ds_cmip_raw_coarse = of.open_nc_file(
                    sample_path_coarse, verbose=False)
                if 'sample_id' in gut.get_dims(ds_cmip_raw_fine):
                    ds_cmip_raw_fine = ds_cmip_raw_fine.mean(dim='sample_id')
                ds_cmip_raw = ds_cmip_raw_fine if savepath_dict == savepath_dict_fine else ds_cmip_raw_coarse
                ds_cmip = gut.translate_cmip2era5(ds_cmip_raw)

                savepath_cutout = f'{config['data_dir']}/{country_name}/{config['data']['CMIP6']}_{gcm_str}_{fine_res}_{tr_str}.nc' if savepath_dict == savepath_dict_fine else f'{config['data_dir']}/{country_name}/{config['data']['CMIP6']}_{gcm_str}_{coarse_res}_{tr_str}.nc'

                ds_cutout = cfu.compute_cutouts(ds=ds_cmip)

                cutout_germany = at.Cutout(
                    savepath_cutout,
                    data=ds_cutout)
                cutout_germany.data.load()

                cf_dict_cmip = cfu.compute_cf_dict(
                    # sources=sources,
                    cutout_country=cutout_germany,
                    config=config,
                    country_name=country_name,
                    correct_qm=False)

                fut.save_np_dict(cf_dict_cmip, savepath_dict,)
# %%
