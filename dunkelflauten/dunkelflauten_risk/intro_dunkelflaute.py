# %%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
import cartopy.crs as ccrs
import atlite as at

from importlib import reload
import os
import yaml
import xarray as xr
import numpy as np
import pandas as pd
# %%
if os.getenv("HOME") == '/home/ludwig/fstrnad80':
    cmip6_dir = "/mnt/lustre/work/ludwig/shared_datasets/CMIP6/"
    data_dir = f'{cmip6_dir}/downscaling/'
    era5_dir = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/Europe"
    with open('../config_cluster.yaml', 'r') as file:
        config = yaml.safe_load(file)
    plot_dir = "/mnt/lustre/home/ludwig/fstrnad80/plots/dunkelflauten/methods/"

else:
    plot_dir = "/home/strnad/plots/dunkelflauten/downscaling_cmip6/"
    data_dir = "/home/strnad/data/CMIP6/downscaling/"
    cmip6_dir = "/home/strnad/data/CMIP6/"
    era5_dir = "/home/strnad/data/climate_data/Europe"
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

# 1. Load the data
gs = .25
country_name = 'Germany'
gs_era5 = 0.25
tr_str = '1980-01-01_2025-01-01'
cf_dict_path_era5 = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{gs_era5}_{tr_str}.npy'

cf_dict = fut.load_np_dict(cf_dict_path_era5)
ts_offwind = cf_dict['offwind']['ts']
ts_onwind = cf_dict['onwind']['ts']
ts_solar = cf_dict['solar']['ts']

# %%
country_name = "Germany"
savepath = f'{config['data_dir']}/{country_name}/cutouts/{config['data']['ERA5']}_cutout_{country_name}_{gs}.nc'
cutout_germany = at.Cutout(savepath)
# %%
# cutout_germany.data.load()
lon_range, lat_range = sput.get_lon_lat_range(cutout_germany.data)

# %%
# Plot a Figure that summarizes the process how to get the dunkelflauten data

reload(tu)
df_type = 'all'
ts_df = cf_dict[df_type]['ts']
num_years = tu.count_unique_years(ts_df)
short_time_range = ['2024-01-01', '2024-12-31']

hourly_res = tu.get_frequency_resolution_hours(ts_df)

num_hours = 48
window = int(num_hours / hourly_res)  # 8*6 = 48 hours
ts_df_mean = tu.rolling_timemean(ts_df, window=window)

threshold = 0.06
ts_evs = tu.compute_evs(ts_df_mean,
                        threshold=threshold,
                        threshold_type='lower',
                        get_mask=False)

tps_dfl, len_dfl, n_dfl, avg_len_dfl = tu.analyze_binary_event_series(ts_evs)
dfl_per_year = tu.count_time_points(tps_dfl)

dfl_per_month = tu.count_time_points(tps_dfl, counter='month')

dfl_per_week = tu.count_time_points(tps_dfl, counter='week')


ts_df_mean_short = tu.get_time_range_data(ts_df_mean,
                                          time_range=short_time_range)
tps_dfl_short = tu.get_time_range_data(tps_dfl,
                                       time_range=short_time_range)
tps_dfl_short_values = tu.get_sel_tps_ds(ts_df_mean_short,
                                         tps=tps_dfl_short.time.data)

data_arr = []
timemean = 'dayofyear'
window = 8
yearly_ts_all = 0
for cap_name, cap in cf_dict.items():
    data = cap['ts']
    ts_mean = tu.rolling_timemean(data, window=window)
    yearly_ts = tu.compute_timemean(ts_mean,
                                    timemean=timemean,
                                    groupby=True,
                                    reset_time=True,)
    data_arr.append(yearly_ts)
# %%
reload(gplt)
cells = cutout_germany.grid
plot_grid_dict = dict(
    alpha=0.3,
    edgecolor="k",
    zorder=4,
    aspect="equal",
    facecolor="None",
    transform=ccrs.PlateCarree(),
)

projection = ccrs.PlateCarree()
fig = plt.figure(figsize=(16, 16))
gs = GridSpec(5, 4, figure=fig,
              height_ratios=[0.5, 0.5, 0.5, 1, 1],
              width_ratios=[2, .1, 1, .5],
              wspace=0.5,
              hspace=1.1, )
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[:3, 1:], projection=projection)
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[3, 2:])
ax6 = fig.add_subplot(gs[4:, 0])
ax7 = fig.add_subplot(gs[4:, 2:])

gplt.enumerate_subplots(axs=[ax0, ax1, ax4, ax5, ax6, ax7],
                        pos_y=1.15)

cap_fac = sput.rename_dims(cf_dict['all']['cf'])
ind_matrix = xr.where(
    (cf_dict['solar']['matrix_xr'] + cf_dict['offwind']['matrix_xr']) == 0, np.nan, 1)
im = gplt.plot_map(cap_fac,  # *ind_matrix,
                   ax=ax1,
                   title='Mean Capacity Factors (CFs) (1979-2024)',
                   vmin=0,  vmax=.3,
                   levels=50,
                   tick_step=10,
                   y_title=1.1,
                   cmap='cmo.thermal',
                   label='Capacity Factor (CF) [a.u.]',
                   lon_range=[5, 16],
                   lat_range=[47, 55.5],
                   orientation='vertical',
                   alpha=0.8,
                   lw_borders=1,
                   plot_borders=True)
cells.plot(
    **plot_grid_dict,
    ax=im['ax'],
)

wnd_data = tu.get_time_range_data(cutout_germany.data.wnd10m.mean(
    ["x", "y"]), time_range=short_time_range)
gplt.plot_2d(
    x=wnd_data.time,
    y=wnd_data,
    title='Wind speed at 10m above surface',
    # y_title=1.4,
    ylabel='WS [m/s]',
    xticklabels=[],
    ax=ax0)

influx_data = tu.get_time_range_data(cutout_germany.data.influx.mean([
                                     "x", "y"]), time_range=short_time_range)
gplt.plot_2d(
    x=influx_data.time,
    y=influx_data,
    title='Downward solar radiation',
    ylabel=r'Influx [W/m$^2$]',
    xticklabels=[],
    ax=ax2)

temp_data = tu.get_time_range_data(cutout_germany.data.temperature.mean([
                                   "x", "y"]), time_range=short_time_range)
gplt.plot_2d(
    x=temp_data.time,
    y=temp_data,
    title='Temperature at surface',
    ylabel='Temperature [K]',
    rot=20,
    ax=ax3)


im_ax1 = gplt.plot_2d(x=[ts_df_mean_short.time.data, tps_dfl_short.time.data],
                      y=[ts_df_mean_short, tps_dfl_short_values],
                      ax=ax4,
                      title=f'Detect Dunkelflaute events',
                      label_arr=['48h-mean CF',
                                 'event'],
                      ls_arr=['-', ''],
                      mk_arr=['', r'$\bigstar$'],
                      lw=1.5,
                      color_arr=['tab:blue', 'tab:red'],
                      #  xlabel='Time',
                      ylabel="CF [a.u.]",
                      ylim=(0, 0.46),
                      rot=20,
                      y_title=1.1,
                      )
gplt.plot_hline(ax=im_ax1['ax'], y=threshold, color='r', ls='--',
                label=f'threshold',
                loc='outside',
                lw=1)

# Dunkelflaute per year
reload(tu)
years = tu.get_year(dfl_per_year.time)
gplt.plot_2d(x=years,
             y=dfl_per_year.data,
             ax=ax5,
             #  plot_type='bar',
             title=f'Dunkelflaute events per year',
             y_title=1.1,
             ls_arr=['-'],
             lw=2,
             color='tab:red',
             mk_arr=[''],
             #  xlabel='Year',
             ylabel='No. of events',
             rot=45,
             ylim=(0, 9),
             set_yint=True,
             set_grid=True,
             #  xticks
             #  xticklabels=years[::2],
             )

day_index = data_arr[0].time.data[np.linspace(
    0, len(data_arr[0].time.data) - 31, 12, dtype=int)]
im = gplt.plot_2d(x=data_arr[0].time,
                  y=data_arr,
                  xticks=day_index,
                  xticklabels=tu.months,
                  rot=45,
                  ax=ax6,
                  title=f"Contributions to CF (1979-2024)",
                  y_title=1.1,
                  label_arr=list(cf_dict.keys()),
                  color_arr=['blue', 'tab:blue', 'orange', 'black'],
                  xlabel="Month of Year",
                  ylabel="CF [a.u.]",
                  set_grid=True,
                  ylim=(0, .67),
                  loc='outside',
                  )


gplt.plot_2d(x=dfl_per_month.time.data,
             y=dfl_per_month.data/num_years,
             ax=ax7,
             plot_type='bar',
             title=f'Freq. Dunkelflaute/month',
             y_title=1.1,
             label_arr=['Dunkelflaute/month'],
             ls_arr=['-'],
             mk_arr=[''],
             #  xlabel='Year',
             ylabel='No. of events',
             rot=45,
             ylim=(0, 1.05),
             )

savepath = f'{plot_dir}/dunkelflauten_overview.pdf'
gplt.save_fig(savepath=savepath)

# %%
reload(tu)
length = 20
fraction_ones = 0.2
num_ones = int(length * fraction_ones)
arr = np.zeros(length, dtype=int)
arr[:num_ones] = 1
np.random.shuffle(arr)
time = pd.date_range("2000-01-01", periods=length, freq="H")
xr_ts = xr.DataArray(arr, coords={"time": time}, dims="time")
print(xr_ts)

tu.analyze_binary_event_series(xr_ts)

# %%
reload(gplt)
reload(tu)
dfl_per_year = tu.count_tps(len_dfl, counter='year')
len_per_year = tu.compute_timemean(len_dfl, timemean='year')

gplt.plot_2d(y=dfl_per_year.data,
             x=len_per_year.data*hourly_res + 42,
             ls='',
             mk='o',
             ylabel='Dunkelflaute events per year',
             xlabel='Av Length of Dunkelflaute events/year [h]',)

# %%
reload(gplt)
gplt.plot_hist(
    data=(len_dfl.data*hourly_res + 42)/24,
    xlabel='Length of Dunkelflaute events [days]',
    ylabel='Frequency',
    density=True,
    xlim=(1.9,8),
    # mk='o',
    # nbins=11,
)