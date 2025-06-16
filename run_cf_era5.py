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
import geoutils.countries.countries as cnt
import geoutils.countries.capacities as cap
import geoutils.cutouts.prepare_cutout as pc

import yaml
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# %%
plot_dir = "/home/strnad/plots/dunkelflauten/downscaling_cmip6/"
data_dir = "/home/strnad/data/CMIP6/downscaling/"
cmip6_dir = "/home/strnad/data/CMIP6/"
era5_dir = "/home/strnad/data/climate_data/Europe"
# %%
reload(of)
reload(gut)
reload(cfu)

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
    ('2010-01-01', '2015-01-01'),
    ('1980-01-01', '2025-01-01'),]

variables = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'surface_solar_radiation_downwards',
    "toa_incident_solar_radiation",
]


# ERA5 data
files_gt = []
files_coarse = []
hourly_res = 6
for variable in variables:
    files_gt.append(
        f'{era5_dir}/{country_name}_nn/{fine_res}/{variable}_{fine_res}_{hourly_res}h.nc')
    files_coarse.append(
        f'{era5_dir}/{country_name}_nn/{coarse_res}/{variable}_{coarse_res}_{hourly_res}h.nc')

ds_era5_fine = of.open_nc_file(files_gt).load()
ds_era5_coarse = of.open_nc_file(files_coarse).load()

# %%
reload(cfu)
time_ranges = tr_historical

for (start_date, end_date) in time_ranges:
    tr_str = f'{start_date}_{end_date}'

    savepath_dict_fine = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{fine_res}_{tr_str}.npy'
    savepath_dict_coarse = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{coarse_res}_{tr_str}.npy'

    gut.myprint(f'time range {tr_str}')
    for savepath_dict in [savepath_dict_fine,
                          savepath_dict_coarse]:
        if fut.exist_file(savepath_dict):
            ds_era5_tr = tu.get_time_range_data(
                ds_era5_fine, time_range=(start_date, end_date))

            savepath_cutout = f'{config['data_dir']}/{country_name}/{config['data']['ERA5']}_{fine_res}_{tr_str}.nc' if savepath_dict == savepath_dict_fine else f'{config['data_dir']}/{country_name}/{config['data']['ERA5']}_{coarse_res}_{tr_str}.nc'

            ds_cutout = cfu.compute_cutouts(ds=ds_era5_tr)

            cutout_germany = at.Cutout(
                savepath_cutout,
                data=ds_cutout)
            cutout_germany.data.load()

            cf_dict_cmip = cfu.compute_cf_dict(
                # sources=sources,
                cutout_country=cutout_germany,
                config=config,
                country_name=country_name,
                correct_qm=True)

            fut.save_np_dict(cf_dict_cmip, savepath_dict,)
# %%
