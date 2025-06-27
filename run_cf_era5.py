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
fine_res = 0.25
coarse_res = 1.0
N = 3

tr_historical = [
    ('2023-01-01', '2025-01-01'),
    # ('1980-01-01', '2025-01-01'),
    # ('1980-01-01', '1990-01-01'),
    # ('1990-01-01', '2000-01-01'),
    # ('2000-01-01', '2010-01-01'),
    # ('2010-01-01', '2015-01-01'),
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
ds_era5_coarse = of.open_nc_file(files_coarse).load()
ds_era5_daily = tu.compute_timemean(ds=ds_era5_fine, timemean='day')
# %%
tr = '2023-01-01_2025-01-01'
N = 10
folder_name = f'{tr}_N{N}'
use_log = False
ds_era5_ds_path = f'/mnt/lustre/home/ludwig/fstrnad80/data/dunkelflauten/downscaling/eval_with_gt/{folder_name}/'
ds_era5_file = f'{ds_era5_ds_path}/samples_era5_{tr}_{fine_res}_log_{use_log}.nc'
ds_era5_ds_samples = of.open_nc_file(ds_era5_file)
ds_era5_ds_samples.load()
# ds_era5_ds = ds_era5_ds_samples.mean(dim='sample_id')
ds_era5_ds = ds_era5_ds_samples.sel(sample_id=0)

ds_era5_file_bc = f'{ds_era5_ds_path}/samples_era5_{tr}_{fine_res}_log_{use_log}_bc.nc'
ds_era5_ds_bc_samples = of.open_nc_file(ds_era5_file_bc)
ds_era5_ds_bc_samples.load()
# %%
reload(cfu)
time_ranges = tr_historical
overwrite = True  # Set to True to overwrite existing files
for (start_date, end_date) in time_ranges:
    tr_str = f'{start_date}_{end_date}'

    savepath_dict_fine = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{fine_res}_{tr_str}.npy'
    savepath_dict_fine_ds = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{fine_res}_{tr_str}_dws.npy'
    savepath_dict_fine_ds_bc = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{fine_res}_{tr_str}_dws_bc.npy'
    savepath_dict_coarse = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{coarse_res}_{tr_str}.npy'
    savepath_dict_daily = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_daily_{fine_res}_{tr_str}.npy'

    savepaths_gt = [savepath_dict_fine,
                    savepath_dict_coarse,
                    savepath_dict_daily]
    savepaths_ds = [savepath_dict_fine_ds,
                    savepath_dict_fine_ds_bc] + savepaths_gt

    savepaths = savepaths_gt if tr_str != '2023-01-01_2025-01-01' else savepaths_ds

    gut.myprint(f'time range {tr_str}')
    for savepath_dict in savepaths:
        if not fut.exist_file(savepath_dict) or overwrite:
            if savepath_dict == savepath_dict_fine:
                ds_era5_tr = tu.get_time_range_data(
                    ds_era5_fine, time_range=(start_date, end_date))
            elif savepath_dict == savepath_dict_coarse:
                ds_era5_tr = tu.get_time_range_data(
                    ds_era5_coarse, time_range=(start_date, end_date))
            elif savepath_dict == savepath_dict_fine_ds:
                ds_era5_tr = tu.get_time_range_data(
                    ds_era5_ds, time_range=(start_date, end_date))
            elif savepath_dict == savepath_dict_fine_ds_bc:
                ds_era5_tr = tu.get_time_range_data(
                    ds_era5_ds_bc_samples, time_range=(start_date, end_date))
            else:
                ds_era5_tr = tu.get_time_range_data(
                    ds_era5_daily, time_range=(start_date, end_date))
            savepath_cutout = f'{config['data_dir']}/{country_name}/{config['data']['ERA5']}_cutout_tmp.nc'

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
                correct_qm=True if tr_str != '2023-01-01_2025-01-01' else False,
                )

            fut.save_np_dict(cf_dict_cmip, savepath_dict)
            # break
# %%
