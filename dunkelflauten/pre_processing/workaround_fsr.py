# %%
# This file is used to create pseudo-fsr data for CMIP6 data that is basically old ERA5 data. This is justified, as the fsr variable is only very marginally changing.
# %%
"""
Variables to analzye are the following:
2m_temperature t2m
forecast_surface_roughness fsr -> roughness
surface_net_solar_radiation ssr -> albedo
surface_solar_radiation_downwards ssrd -> albedo, influx_diffuse (=ssrd-fdir)
toa_incident_solar_radiation tisr -> influx_toa
total_sky_direct_solar_radiation_at_surface fdir -> influx_direct
"""
import xarray as xr
import geoutils.preprocessing.open_nc_file as of
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
from importlib import reload
import yaml


# %%


def create_means(
        fsr_variable="forecast_surface_roughness",
        country_name='Germany',
        hourly_res=6.0,
        means_folder=f'./variable_means/',
        override=True):
    reload(of)
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    for gs in [0.25, 0.5, 1.0, 2.0]:
        fsr_file = f"{config['europe_dir']}/{country_name}_av/{gs}/{fsr_variable}_{gs}_{hourly_res}h.nc"
        fsr_data_path = f'{means_folder}/mean_{fsr_variable}_{country_name}_{gs}.nc'
        if not fut.exist_file(fsr_data_path) or override:
            fut.assert_file_exists(fsr_file)
            fsr_data_raw = of.open_nc_file(fsr_file)
            mean_fsr = fsr_data_raw.mean(dim='time').rename(
                {fsr_variable: 'fsr_mean'})
            std_fsr = fsr_data_raw.std(dim='time').rename(
                {fsr_variable: 'fsr_std'})

            fsr_data = xr.merge([mean_fsr, std_fsr])
            fut.save_ds(fsr_data, fsr_data_path)


def create_pseudo_fsr_data(
    fsr_variable="forecast_surface_roughness",
    country_name='Germany',
    means_folder=f'./variable_means/',
    gs=0.5,
    hourly_res=6,
    start_date='2023-01-01',
    end_date='2023-12-31',
    data_array_like=None,
):
    reload(of)
    if data_array_like is not None:
        times = data_array_like.time
        gs = sput.get_grid_step(data_array_like)[0]

    else:
        times = tu.get_dates_in_range(start_date=start_date,
                                      end_date=end_date,
                                      freq=f'{int(24/hourly_res)}h')
        gs = float(gs)
    fsr_data_path = f'{means_folder}/mean_{fsr_variable}_{country_name}_{gs}.nc'
    fsr_means = of.open_nc_file(fsr_data_path)
    fsr_mean = fsr_means['fsr_mean']
    fsr_std = fsr_means['fsr_std']


    mean_ts_arr = fsr_mean.expand_dims(time=times)

    return mean_ts_arr
# %%
