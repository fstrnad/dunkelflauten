# %%
import cf_utils as cfu
import geoutils.preprocessing.open_nc_file as of
import geoutils.utils.time_utils as tu
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
    era5_dir = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/Europe"
    with open('./config_cluster.yaml', 'r') as file:
        config = yaml.safe_load(file)
else:
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
fine_res = 0.25
coarse_res = 1.0
N = 3

tr_historical = [
    ('2023-01-01', '2025-01-01'),
    ]

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


# %%
reload(cfu)
time_ranges = tr_historical
overwrite = True  # Set to True to overwrite existing files
for (start_date, end_date) in time_ranges:
    tr_str = f'{start_date}_{end_date}'

    gut.myprint(f'time range {tr_str}')
    ds_era5_tr = tu.get_time_range_data(
                    ds_era5_fine, time_range=(start_date, end_date))
    savepath_cutout = f'{config['data_dir']}/{country_name}/cutouts/{config['data']['ERA5']}_cutout_{country_name}_{fine_res}.nc'

    ds_cutout = cfu.compute_cutouts(ds=ds_era5_tr)

    cutout_germany = at.Cutout(
        savepath_cutout,
        data=ds_cutout)
    cutout_germany

    fut.save_ds(ds_cutout, savepath_cutout)

    # break
# %%
