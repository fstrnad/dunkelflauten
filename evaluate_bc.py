# %%
import geoutils.utils.statistic_utils as sut
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import geoutils.geodata.wind_dataset as wds
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.plotting.plots as gplt
import geoutils.preprocessing.open_nc_file as of
import sda_atmos.eval.sample_data as sda
import sda_atmos.eval.util as sda_utils
import sda_atmos.training.dataset as tds

from pathlib import Path
from importlib import reload
import yaml

reload(sda)
plot_dir = "/home/strnad/plots/dunkelflauten/bias_correction/"
data_dir = "/home/strnad/data/climate_data/Europe"
cmip6_dir = '/home/strnad/data/CMIP6/'
era5_dir = data_dir

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
# %%
var_cmip6 = 'vas'
var_era5 = gut.cmip2era5_dict[var_cmip6]
hourly_res = 6
country_name = 'Germany'
gcm = 'MPI-ESM1-2-HR'
scenario = 'ssp245'
gcm_str = f'{gcm}_{scenario}'
gs_cmip = 1.0
gs_era5 = gs_cmip
cmip6_folder = f"{cmip6_dir}/{country_name}/{gs_cmip}/"

# %%
variable_dict = {
    '2m_temperature': dict(
        cmap='RdBu_r',
        vmin=-10, vmax=35,
        label='Temperature [°C]',
        levels=20,
        vname='Surface Air Temperature',
        offset=-271.15),
    '10m_u_component_of_wind': dict(
        cmap='plasma',
        vmin=-15, vmax=15,
        label='Wind speed [m/s]',
        offset=0,
        levels=20,
        vname='10m U Component of Wind',),
    '10m_v_component_of_wind': dict(
        cmap='viridis',
        vmin=-15, vmax=15,
        label='Wind speed [m/s]',
        offset=0,
        levels=20,
        vname='10m V Component of Wind',),
    'surface_solar_radiation_downwards': dict(
        cmap='inferno',
        vmin=1e3, vmax=1.5e6,
        label=r'Solar radiation [W/m$^2$]',
        offset=0,
        levels=20,
        yrange=(0, 1e5),
        vname='Surface Solar Radiation Downwards',),
}
# %%


im = gplt.create_multi_plot(nrows=1, ncols=len(variable_dict),
                            figsize=(30, 5)
                            )
for idx, var_era5 in enumerate(variable_dict.keys()):
    var_cmip6 = gut.era52cmip_dict[var_era5]
    filename = f'{gcm}_{scenario}_{var_cmip6}_{gs_cmip}.nc'
    # always take 1st run in case of multiple
    file = f'{cmip6_folder}/{filename}'
    ds_cmip_scenario = of.open_nc_file(file, compat='override',
                                       check_vars=True
                                       )
    ds_cmip_hist = of.open_nc_file(file.replace(scenario, 'historical'),
                                   compat='override',
                                   check_vars=True
                                   )
    ds_cmip_scenario = gut.translate_cmip2era5(ds_cmip_scenario)
    ds_cmip_hist = gut.translate_cmip2era5(ds_cmip_hist)
    era5_file = f'{era5_dir}/{country_name}_nn/{gs_era5}/{var_era5}_{gs_era5}_{hourly_res}h.nc'
    ds_era5 = of.open_nc_file(era5_file)
    ds_era5 = tu.compute_timemean(ds=ds_era5,
                                  timemean='day')
    ds_era5, ds_cmip_hist = tu.equalize_time_points(
        ds_era5, ds_cmip_hist
    )

    cmip6_folder_bc = f"{cmip6_dir}/{country_name}/{gs_cmip}_bc/"
    file_qm = f'{cmip6_folder_bc}/qm_{filename}'
    file_qdm = f'{cmip6_folder_bc}/qdm_{filename}'

    ds_qdm = of.open_nc_file(file_qdm)
    ds_dict = {
        'CMIP6 future': ds_cmip_scenario,
        'CMIP6 historical': ds_cmip_hist,
        'ERA5': ds_era5,
        # 'CMIP6 QM future': ds_qm,
        'CMIP6 QDM future': ds_qdm
    }
    tr_distr = tu.tps2str(tu.get_time_range(ds_qdm),
                          h=False, full_str=True)

    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        offset = variable_dict[var_era5]['offset']
        plot_data = ds[var_era5].values.flatten() + offset

        gplt.plot_hist(plot_data,
                       ax=im['ax'][idx],
                       title=var_era5,
                       color=gplt.colors[i],
                       label=ds_type if idx == len(variable_dict)-1 else None,
                       nbins=100,
                       lw=1.5,
                       alpha=0.8,
                       set_yaxis=False,
                       vertical_title=f'{gcm} {scenario}' if idx == 0 else None,
                    #    ylim=variable_dict[var_era5]['yrange'] if var_era5 == 'surface_solar_radiation_downwards' else None,
                       ylabel='Density',
                       density=True,
                       xlabel=variable_dict[var_era5]['label'],
                       xlim=(variable_dict[var_era5]['vmin'],
                             variable_dict[var_era5]['vmax']),
                       )

savepath = f'{plot_dir}/qdm_{gcm_str}_all_vars_{tr_distr}.png'
gplt.save_fig(savepath=savepath)

# %%
