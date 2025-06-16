# %%
import sys
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
ssps = ['historical', 'ssp245', 'ssp585']

gcm_ssp_cf_dict = {}
for gcm in gcms:
    ssp_cf_dict = {}
    for ssp in ssps:
        gcm_str = f'{gcm}_{ssp}'
        cf_tr_dict = {}
        time_ranges = tr_ssp if ssp != 'historical' else tr_historical
        for tr_idx, (start_date, end_date) in enumerate(time_ranges):
            start_date, end_date = time_ranges[tr_idx]
            tr_str = f'{start_date}_{end_date}'

            reload(cfu)
            savepath_dict = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{fine_res}_{tr_str}.npy'
            if fut.exist_file(savepath_dict):
                cf_dict_cmip = fut.load_np_dict(savepath_dict)
                cf_tr_dict[tr_str] = cf_dict_cmip
            else:
                print(f'{gcm} {ssp} {tr_str} not found')
        ssp_cf_dict[ssp] = cf_tr_dict
    gcm_ssp_cf_dict[gcm] = ssp_cf_dict
# %%
# For comparison with ERA5
gs_era5 = 0.25
cf_dict_path_era5 = f'{config['data_dir']}/{country_name}/era5/cf_dict_{gs_era5}.npy'

cf_dict_era5 = fut.load_np_dict(cf_dict_path_era5)
# %%
reload(gplt)
ssp = 'historical'
gcm_str = f'{gcm}_{ssp}'
cf_tr_dict = ssp_cf_dict[ssp]
ncols = 3
nrows = len(cf_tr_dict)
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.2)
im_diff = gplt.create_multi_plot(nrows=nrows-1, ncols=ncols,
                                 projection='PlateCarree',
                                 wspace=0.2)
cf_0_dict = cf_tr_dict[list(cf_tr_dict.keys())[0]]
sd, ed = tu.get_time_range(
    cf_0_dict['wind']['ts'], asstr=True, m=False, d=False)
for tr_idx, (tr_str, cf_dict_cmip) in enumerate(cf_tr_dict.items()):
    sd_tr, ed_tr = tu.get_time_range(cf_dict_cmip['wind']['ts'], asstr=True)
    vertical_title = f'{gcm} - {ssp} \n({sd_tr} - {ed_tr})'

    for idx, (sname, source) in enumerate(cf_dict_cmip.items()):
        if sname != 'wind' and sname != 'all':
            cap_fac = source['cf']
            cf_0 = cf_0_dict[sname]['cf']
            gplt.plot_map(cap_fac,
                          ax=im['ax'][tr_idx*ncols + idx],
                          title=sname if tr_idx == 0 else None,
                          vertical_title=vertical_title if idx == 0 else None,
                          cmap='cmo.solar',
                          vmin=0,
                          vmax=.15 if sname == 'solar' else 0.4,
                          label='Capacity Factor [a.u.]' if tr_idx == nrows -
                          1 else None,
                          plot_borders=True)
            # Plot the difference to the first time period
            if tr_idx > 0:
                vmin = -0.03 if sname == 'solar' else -0.05
                gplt.plot_map(cap_fac - cf_0,
                              ax=im_diff['ax'][(tr_idx-1)*ncols + idx],
                              title=sname if tr_idx == 1 else None,
                              vertical_title=vertical_title if idx == 0 else None,
                              vmin=vmin,
                              vmax=-vmin,
                              cmap='cmo.tarn_r',
                              centercolor='white',
                              label=f'Difference CF ({sd} - {ed}) [a.u.]' if tr_idx == nrows - 1
                              else None,
                              plot_borders=True)


savepath = f"{config['plot_dir']}/capacities_cmip6/cf_types_full_{gcm_str}_{fine_res}.png"
gplt.save_fig(savepath, fig=im['fig'])
savepath_diff = f"{config['plot_dir']}/capacities_cmip6/cf_types_diff_{gcm_str}_{fine_res}.png"
gplt.save_fig(savepath_diff, fig=im_diff['fig'])

# %%
# Plot Capacity Factors
timemean = 'dayofyear'
window = 8
yearly_ts_all = 0
for ssp, cf_tr_dict in ssp_cf_dict.items():
    gcm_str = f'{gcm}_{ssp}'
    for tr_str, cf_dict_cmip in cf_tr_dict.items():
        data_arr = []
        for cap_name, cap in cf_dict_cmip.items():
            if cap_name == 'wind':
                continue
            data = cap['ts']
            ts_mean = tu.rolling_timemean(data, window=window)
            yearly_ts = tu.compute_timemean(ts_mean,
                                            timemean=timemean,
                                            groupby=True,
                                            reset_time=True,)
            data_arr.append(yearly_ts)

        colors = ['blue', 'tab:blue', 'orange', 'black']

        sd, ed = tu.get_time_range(data, asstr=True)
        im = gplt.plot_2d(x=data_arr[0].time,
                          y=data_arr,
                          title=f"Contributions to capacity factor ({sd} - {ed})",
                          vertical_title=f'{gcm} - {ssp}',
                          label_arr=list(cf_dict_cmip.keys()) + ['All'],
                          color_arr=colors,
                          xlabel="Day of Year",
                          ylabel="Capacity Factor [a.u.]",
                          set_grid=True,
                          ylim=(0, .9),
                          loc='outside',
                          )
        savepath = f"{config['plot_dir']}/CMIP6/{country_name}_{gcm_str}_{tr_str}_{gs_dws}_capacity_factor_per_day.png"
        gplt.save_fig(savepath, fig=im['fig'])

# %%
# plot the single time_series
reload(tu)
reload(gplt)
subdevide = 4  # 4 tps per day
window = int(365*subdevide)
yearly_ts_all = 0
x_lin_reg = np.linspace(0, 1, 100)

time_range = None
for ssp, cf_tr_dict in ssp_cf_dict.items():
    gcm_str = f'{gcm}_{ssp}'
    im = None
    for tr_str, cf_dict_cmip in cf_tr_dict.items():
        data_arr = []
        trend_arr = []

        for cap_name, cap in cf_dict_cmip.items():
            if cap_name == 'wind':
                continue
            da = cap['ts']
            if window > 1:
                da = tu.rolling_timemean(da, window=window,
                                         center=False,
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
                          ax=im['ax'] if im is not None else None,
                          #  power_pv.values,
                          title=f"{gcm} {ssp} CF",
                          label_arr=list(cf_dict_cmip.keys()),
                          color_arr=colors,
                          xlabel="Year",
                          ylabel="Capacity Factor [a.u.]",
                          set_grid=True,
                          ylim=(0, .6),
                          loc='outside',
                          )
        im = gplt.plot_2d(x=trend_arr[0].time,
                          y=trend_arr,
                          ax=im['ax'],
                          color_arr=['blue', 'tab:blue', 'orange', 'black'],
                          set_axis=False,
                          )

    savepath = f"{config['plot_dir']}/capacities_cmip6/{gcm}_{ssp}_{gs_dws}_capacity_factor_time_period.png"
    gplt.save_fig(savepath, fig=im['fig'])


