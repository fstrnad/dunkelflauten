# %%
import sys
import cmethods as cm

import capacity_factors.cf_utils as cfu
import geoutils.utils.met_utils as mut
import geoutils.geodata.solar_radiation as sr
import pre_processing.workaround_fsr as wf
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
import geoutils.countries.countries as cnt
import geoutils.countries.capacities as cap
import geoutils.cutouts.prepare_cutout as pc
import os
import yaml
# %%
if os.getenv("HOME") == '/home/ludwig/fstrnad80':
    cmip6_dir = "/mnt/lustre/work/ludwig/shared_datasets/CMIP6/"
    data_dir = f'{cmip6_dir}/downscaling/'
    era5_dir = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/Europe"
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
reload(fut)
variables = ['tas', 'uas', 'vas', 'rsds']
europe_dir = config['europe_dir']

gcm = 'MPI-ESM1-2-HR'
gcm = 'GFDL-ESM4'
time = 'day'


country_name = 'Germany'
gs_dws = 1.0
fine_res = 0.25
N = 3
gcms = [
    'MPI-ESM1-2-HR',
    'GFDL-ESM4',
    'MIROC6',
    'IPSL-CM6A-LR',
    'CanESM5'
]
tr_historical = [
    ('1980-01-01', '1990-01-01'),
    ('1990-01-01', '2000-01-01'),
    ('2000-01-01', '2010-01-01'),
    ('2010-01-01', '2015-01-01')]

tr_ssp = [
    ('2020-01-01', '2030-01-01'),
    ('2030-01-01', '2040-01-01'),
    ('2040-01-01', '2050-01-01'),
    ('2050-01-01', '2060-01-01'),
    ('2060-01-01', '2070-01-01'),
    ('2070-01-01', '2080-01-01'),
    ('2080-01-01', '2090-01-01'),
    ('2090-01-01', '2100-01-01')
]

tr_historical = [('1980-01-01', '2015-01-01')]  # full range
tr_ssp = [('2020-01-01', '2100-01-01')]  # full range

ssps = ['historical', 'ssp245', 'ssp585']

# %%

gs_cmip = 0.25
gs_era5 = 0.25  # needs to be the same for CMIP and ERA5
gs_dws = 1.0  # downscaling input resolution for the DWS data
hourly_res = 6
N = 3
used_variables = ['uas',
                  'vas',
                  'tas',
                  'rsds',
                  ]

era5_files = []
for var_cmip in used_variables:
    var_era5 = gut.cmip2era5_dict[var_cmip]
    gs_era5 = gs_cmip  # needs to be the same for CMIP and ERA5
    era5_file = f'{era5_dir}/{country_name}_nn/{gs_era5}/{var_era5}_{gs_era5}_{hourly_res}h.nc'
    if not fut.exist_file(era5_file):
        gut.myprint(f'File {era5_file} does not exist!')
        continue
    else:
        era5_files.append(era5_file)
        gut.myprint(f'Using file {era5_file} for {var_era5}')

# opens the ERA5 Ground truth data
ds_era5 = of.open_nc_file(era5_files)
ds_era5.load()  # load the data into memory
# %%
overwrite = True  # set to True to overwrite existing files
for gcm in gcms:
    # Open the CMIP6 data for the historical period
    start_date, end_date = tr_historical[0]
    tr_str = f'{start_date}_{end_date}'
    gcm_str = f'{gcm}_historical'  # historical ssp
    folder_hist = f'{data_dir}/{gs_dws}/{gcm_str}/{tr_str}_N{N}/'
    hist_file_name = f'samples_{gcm_str}_{tr_str}_{gs_era5}_log_False.nc'
    filename_hist = f'{folder_hist}/{hist_file_name}'
    if not fut.exist_file(filename_hist):
        gut.myprint(f'File {filename_hist} does not exist!')
        continue
    ds_cmip_hist = of.open_nc_file(filename_hist)
    ds_cmip_hist.load()  # load the data into memory
    ds_cmip_hist = ds_cmip_hist.mean(dim='sample_id')
    for ssp in ssps:
        tr = tr_historical if ssp == 'historical' else tr_ssp
        start_date, end_date = tr[0]
        tr_str = f'{start_date}_{end_date}'

        gcm_str = f'{gcm}_{ssp}'
        folder_cmip6 = f'{data_dir}/{gs_dws}/{gcm_str}/{tr_str}_N{N}/'
        cmip6_file = f'samples_{gcm_str}_{tr_str}_{gs_era5}_log_False'
        filename_cmip6 = f'{folder_cmip6}/{cmip6_file}.nc'

        if not fut.exist_file(filename_cmip6):
            gut.myprint(f'File {filename_cmip6} does not exist!')
            continue
        ds_cmip_ssp = of.open_nc_file(filename_cmip6)
        ds_cmip_ssp.load()  # load the data into memory
        ds_cmip_ssp = ds_cmip_ssp.mean(dim='sample_id')
        file_bc = f'{folder_cmip6}/{cmip6_file}_bc.nc'
        if fut.exist_file(file_bc) and not overwrite:
            gut.myprint(f'File {file_bc} already exists!')
        else:
            files = []
            for var_cmip in used_variables:
                print(f'Processing {var_cmip} {gcm} {ssp}')
                var_era5 = gut.cmip2era5_dict[var_cmip]
                obs_data = ds_era5[var_era5]
                hist_mod_data = ds_cmip_hist[var_era5]
                fut_mod_data = ds_cmip_ssp[var_era5]
                obs_data, hist_mod_data = tu.equalize_time_points(
                    obs_data, hist_mod_data
                )

                qm_results = cm.adjust(
                    method="quantile_delta_mapping" if var_cmip in [
                        'tas', 'rsds'] else "quantile_mapping",
                    obs=obs_data,
                    simh=hist_mod_data,
                    simp=fut_mod_data,
                    n_quantiles=10000,
                    kind="+",
                )
                files.append(qm_results)
            qm_results = xr.merge(files)
            fut.save_ds(ds=qm_results,
                        filepath=file_bc,)
            # sys.exit()
# %%
