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
import yaml
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)


plot_dir = "/home/strnad/plots/dunkelflauten/paper_plots/"
data_dir = "/home/strnad/data/dunkelflauten/downscaling/eval_with_gt/2023-01-01_2023-01-30_L120_N3/"
cmip6_dir = "/home/strnad/data/CMIP6/"
era5_dir = "/home/strnad/data/climate_data/Europe"
# %%
# Plot a Figure that summarizes the process how to get the dunkelflauten data

# 1. Load the data
gs = .5
country_name = "Germany"
cf_dict_path = f'{config['data_dir']}/{country_name}/era5/cf_dict_{gs}.npy'

cf_dict = fut.load_np_dict(cf_dict_path)
ts_offwind = cf_dict['offwind']['ts']
ts_onwind = cf_dict['onwind']['ts']
ts_solar = cf_dict['solar']['ts']

# %%
country_name = "Germany"
savepath = f'{config['data_dir']}/{country_name}/{config['data'][gs]}'
cutout_germany = at.Cutout(savepath)
# %%
# cutout_germany.data.load()
lon_range, lat_range = sput.get_lon_lat_range(cutout_germany.data)

# %%
# Plot the capacify factor time series for whole germany
df_type = 'all'
ts_df = cf_dict[df_type]['ts']
short_time_range = ['2015-01-01', '2016-01-30']

hourly_res = tu.get_frequency_resolution_hours(ts_df)

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

dfl_per_week = tu.count_time_points(tps_dfl, counter='week')


ts_df_mean = tu.get_time_range_data(ts_df_mean,
                                    time_range=short_time_range)
tps_dfl = tu.get_time_range_data(tps_dfl,
                                 time_range=short_time_range)

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
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(5, 4, figure=fig,
              height_ratios=[0.5, 0.5, 0.5, 1, 1],
              width_ratios=[2, .2, 1, .5],
              wspace=0.4,
              hspace=1.1, )
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[:3, 1:], projection=projection)
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[3, 2:])
ax6 = fig.add_subplot(gs[4:, 0])
ax7 = fig.add_subplot(gs[4:, 2:])

gplt.enumerate_subplots(axs=[ax0, ax1, ax4, ax5, ax6, ax7],)
cap_fac = sput.rename_dims(cf_dict['all']['cf'])
ind_matrix = xr.where(
    (cf_dict['solar']['matrix_xr'] + cf_dict['offwind']['matrix_xr']) == 0, np.nan, 1)
im = gplt.plot_map(cap_fac,  # *ind_matrix,
                   ax=ax1,
                   title='Capacity Factors Germany (1979-2024)',
                   vmin=0,  vmax=.3,
                   #   projection=projection,
                   label='Capacity Factor',
                   lon_range=[4, 16],
                   lat_range=[46, 56],
                   orientation='vertical',
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

influx_data = tu.get_time_range_data(cutout_germany.data.influx_direct.mean([
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


im_ax1 = gplt.plot_2d(x=[ts_df_mean.time.data, tps_dfl.time.data],
                      y=[ts_df_mean, tps_dfl],
                      ax=ax4,
                      title=f'Detect Dunkelflaute events',
                      label_arr=['CF',
                                 'event'],
                      ls_arr=['-', ''],
                      mk_arr=['', 'x'],
                      lw=1.5,
                      color_arr=['tab:blue', 'tab:red'],
                      #  xlabel='Time',
                      ylabel="Capacity Factor [a.u.]",
                      ylim=(0, 0.6),
                      rot=20,
                      )
gplt.plot_hline(ax=im_ax1['ax'], y=threshold, color='r', ls='--',
                label=f'Threshold',
                loc='outside',
                lw=1)

# Dunkelflaute per year
reload(tu)
years = tu.get_year(dfl_per_year.time)
gplt.plot_2d(x=years,
             y=dfl_per_year.data,
             ax=ax5,
            #  plot_type='bar',
             title=f'Dunkelflaute per year',
             ls_arr=['-'],
             lw=2,
             color='tab:red',
             mk_arr=[''],
             #  xlabel='Year',
             ylabel='No. of events',
             rot=90,
             ylim=(0, 9),
             set_yint=True,
            #  xticks
            #  xticklabels=years[::2],
             )


im = gplt.plot_2d(x=data_arr[0].time,
                  y=data_arr,
                  ax=ax6,
                  title=f"Contributions to capacity factor (1979-2024)",
                  label_arr=list(cf_dict.keys()) + ['All'],
                  color_arr=['blue', 'tab:blue', 'c', 'orange', 'black'],
                  xlabel="Day of Year",
                  ylabel="Capacity Factor [a.u.]",
                  set_grid=True,
                  ylim=(0, .7),
                  loc='outside',
                  )


gplt.plot_2d(x=dfl_per_month.time.data,
             y=dfl_per_month.data,
             ax=ax7,
             plot_type='bar',
             title=f'Dunkelflaute per month',
             label_arr=['Dunkelflaute per month'],
             ls_arr=['-'],
             mk_arr=[''],
             #  xlabel='Year',
             ylabel='No. of events',
             rot=90,
             ylim=(0, 26),
             )

savepath = f'{plot_dir}/dunkelflauten_overview.png'
gplt.save_fig(savepath=savepath)
# %%
# Risk Assessment

reload(gplt)
# %%
# Local dunkelflauten Germany
reload(gplt)

w_on_solar = cf_dict['onwind']['weight'] + cf_dict['solar']['weight']
cf_onwind_solar = (cf_dict['onwind']['weight'] * cf_dict['onwind']['cf_ts'] + \
    cf_dict['solar']['weight'] * cf_dict['solar']['cf_ts']) * 1/w_on_solar
# %%
reload(tu)
cf_onwind_solar = sput.rename_dims(cf_onwind_solar)
num_hours = 48
window = int(num_hours / hourly_res)  # 8*6 = 48 hours
cf_ts_mean = tu.rolling_timemean(cf_onwind_solar, window=window)
threshold = 0.02
df_local_onwind = tu.compute_evs(cf_ts_mean,
                                 threshold=threshold,
                                 threshold_type='lower',
                                 max_rel_share=0.01,
                                 get_mask=False)
# %%
# ERA5
reload(tu)
reload(gplt)
time_ranges = tu.split_time_by_year_span(df_local_onwind, span_years=10)
im = gplt.create_multi_plot(nrows=1, ncols=len(time_ranges),
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
                  vmin=0,
                  vmax=20,
                  label='No. of Dunkelflauten / Year',
                  title=f'{tr[0]} - {tr[1]}',
                  vertical_title='ERA5' if idx == 0 else None,
                  cmap='Reds',
                  leftcolor='white',
                  levels=10,
                  )

savepath = f'{plot_dir}/dunkelflauten_local_{country_name}_era5.png'
gplt.save_fig(savepath, fig=im['fig'])

# %%
data_arr = []
subdevide = 1
window = int(365/subdevide)
yearly_ts_all = 0
trend_arr = []
x_lin_reg = np.linspace(0, 1, 100)

time_range = None
for cap_name, cap in cf_dict.items():
    if cap_name == 'wind':
        continue
    da = cap['ts']
    if window > 1:
        da = tu.rolling_timemean(da, window=window, center=False,
                                 )
    if time_range is not None:
        da = tu.get_time_range_data(da, time_range)
    time = da.time
    data_arr.append(da)
    # compute the trend
    num_tps = (time.values - time.values[0]
               ).astype('timedelta64[h]').astype(int)
    coeffs = np.polyfit(num_tps, da.values, deg=1)
    trend = np.polyval(coeffs, num_tps)
    trend = xr.DataArray(trend, coords={'time': da.time}, dims='time')
    trend_arr.append(trend)

sd, ed = tu.get_time_range(data, asstr=True)
im = gplt.plot_2d(x=data_arr[0].time,
                  y=data_arr,
                  #  power_pv.values,
                  title=f"Mean Cap. Factor {sd} - {ed}",
                  vertical_title=f'ERA5',
                  label_arr=list(cf_dict.keys()),
                  color_arr=['blue', 'tab:blue', 'orange', 'black'],
                #   xlabel="Day of Year",
                ylabel="Capacity Factor [a.u.]",
                  set_grid=True,
                  #   ylim=(0.1, .5),
                  )
im = gplt.plot_2d(x=trend_arr[0].time,
                  y=trend_arr,
                  ax=im['ax'],
                  color_arr=['blue', 'tab:blue', 'orange', 'black'],
                  )

savepath = f'{plot_dir}/{country_name}_era5_capacity_factor_time_period.png'
gplt.save_fig(savepath, fig=im['fig'])