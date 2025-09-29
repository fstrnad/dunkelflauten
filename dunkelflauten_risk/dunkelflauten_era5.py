# %%
import geoutils.utils.statistic_utils as sut
import capacity_factors.cf_utils as cfu
from scipy import stats
from matplotlib.gridspec import GridSpec
from cartopy.crs import PlateCarree as plate
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
country_name = 'Germany'
gs_era5 = 0.25
tr_str = '1980-01-01_2025-01-01'
cf_dict_path_era5 = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{gs_era5}_{tr_str}.npy'

cf_dict_era5 = fut.load_np_dict(cf_dict_path_era5)
# %%
ts_offwind = cf_dict_era5['offwind']['ts']
ts_onwind = cf_dict_era5['onwind']['ts']
ts_solar = cf_dict_era5['solar']['ts']


# %%
reload(tu)
df_type = 'all'
time_range = ['1979-01-01', '2024-12-31']
ts_df = tu.get_time_range_data(cf_dict_era5[df_type]['ts'],
                               time_range=time_range)
hourly_res = tu.get_frequency_resolution_hours(ts_df)
num_years = tu.count_unique_years(ts_df)
num_hours = 48
window = int(num_hours / hourly_res)  # 8*6 = 48 hours
ts_df_mean = tu.rolling_timemean(ts_df, window=window)

threshold = 0.06
ts_evs = tu.compute_evs(ts_df_mean,
                        threshold=threshold,
                        threshold_type='lower',
                        get_mask=False)
tps_dfl_raw = ts_df_mean.where(ts_evs == 1).dropna('time')
min_consecutive = 1
tps_dfl = tu.find_consecutive_ones(ts_evs,
                                   min_consecutive=min_consecutive)
tps_dfl = tu.get_sel_tps_ds(ts_df_mean, tps_dfl)
dfl_per_year = tu.count_time_points(tps_dfl)

dfl_per_month = tu.count_time_points(tps_dfl, counter='month')
dfl_per_month = sut.relative_share(dfl_per_month)
dfl_per_week = tu.count_time_points(tps_dfl, counter='week')

reload(gplt)
im = gplt.create_multi_plot(nrows=2, ncols=2,
                            figsize=(22, 10),
                            hspace=0.4,
                            wspace=0.1,
                            )
gplt.plot_2d(x=[ts_df_mean.time.data, tps_dfl.time.data],
             y=[ts_df_mean, tps_dfl],
             ax=im['ax'][0],
             title=f'Identify Dunkelflauten {country_name}',
             label_arr=['CF time series', 'Selected Dunkelflauten   '],
             ls_arr=['-', ''],
             mk_arr=['', 'x'],
             #  xlabel='Time',
             ylabel='CF [a.u]',
             ylim=(0, 0.6),
             rot=90,
             )
gplt.plot_hline(ax=im['ax'][0], y=threshold, color='r', ls='--',
                label=f'Threshold = {threshold}',
                lw=1)

gplt.plot_2d(x=dfl_per_month.time.data,
             y=dfl_per_month.data,
             ax=im['ax'][1],
             plot_type='bar',
             title=f'Dunkelflaute per month {country_name}',
             label_arr=['Dunkelflaute per month'],
             ls_arr=['-'],
             mk_arr=[''],
             #  xlabel='Year',
             ylabel='Rel. Frac. Dunkelflaute events',
             rot=90,
            #  ylim=(0, 26),
             )

gplt.plot_2d(x=tu.get_year(dfl_per_year.time),
             y=dfl_per_year.data,
             ax=im['ax'][2],
             #  plot_type='bar',
             title=f'Dunkelflaute per year {country_name}',
             label_arr=['Dunkelflaute per year'],
             ls_arr=['-'],
             color='tab:blue',
             mk_arr=[''],
             #  xlabel='Year',
             ylabel='Number of Dunkelflaute events',
             rot=90,
             ylim=(0, 9),
             )

gplt.plot_2d(x=dfl_per_week.time.data,
             ax=im['ax'][3],
             y=dfl_per_week.data,
             plot_type='bar',
             title=f'Dunkelflaute per week of year {country_name}',
             label_arr=['Dunkelflaute per week'],
             ls_arr=['-'],
             mk_arr=[''],
             #  xlabel='Year',
             ylabel='Number of Dunkelflaute events',
             rot=90,
             ylim=(0, 9)
             )

savepath = f'{config['plot_dir']}/dunkelflauten_era5/ts_{df_type}_dunkelflauten_era5_{gs_era5}.png'
gplt.save_fig(savepath)

# %%
# Risk Assessment
# Local dunkelflauten Germany
reload(gplt)
reload(cfu)
local_cfs = cfu.combined_cf_maps(cf_dict_era5,
                                 sources=['onwind', 'solar'],)
# %%
reload(tu)
local_cfs = sput.rename_dims(local_cfs)
num_hours = 48
window = int(num_hours / hourly_res)  # 8*6 = 48 hours
cf_ts_mean = tu.rolling_timemean(local_cfs, window=window)
threshold = 0.02
df_local_onwind, mask = tu.compute_evs(cf_ts_mean,
                                       threshold=threshold,
                                       threshold_type='lower',
                                       #    max_rel_share=0.02,
                                       get_mask=True)
# %%
# plot risks ERA5
reload(tu)
reload(gplt)
time_ranges = tu.split_time_by_year_span(df_local_onwind,
                                         start_year=1980,
                                         end_year=2015,
                                         span_years=10)
sd, ed = tu.get_time_range(df_local_onwind, asstr=True)
nrows = 1
im = gplt.create_multi_plot(nrows=1,
                            ncols=int(np.ceil(len(time_ranges)/nrows)),
                            end_idx=len(time_ranges),
                            figsize=(22, 10),
                            hspace=0.4,
                            wspace=0.3,
                            projection='PlateCarree',
                            )
for idx, tr in enumerate(time_ranges):
    tr_df = tu.get_time_range_data(df_local_onwind,
                                   time_range=tr)
    num_years = tu.count_unique_years(tr_df)
    num_dfs_cell = tr_df.sum(dim='time')
    gplt.plot_map(num_dfs_cell/num_years,
                  ax=im['ax'][idx],
                  plot_borders=True,
                  significance_mask=xr.where(mask, 0, 1),
                  vmin=0,
                  vmax=25,
                  label='No. of Dunkelflauten / Year',
                  vertical_title='ERA5' if idx == 0 else '',
                  title=f'{tr[0]} - {tr[1]}',
                  cmap='cmo.amp',
                  leftcolor='white',
                  levels=10,
                  )

savepath = f'{config['plot_dir']}/local_risks/ERA5/dunkelflauten_local_era5_{gs_era5}_{sd}_{ed}.png'
gplt.save_fig(savepath, fig=im['fig'])

# %%
