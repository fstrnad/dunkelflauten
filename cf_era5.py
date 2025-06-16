# %%
import cmethods as cm
import geoutils.bias_correction.quantile_mapping as qm
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
import cf_utils as cfu
import yaml
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
cmip6_dir = "/home/strnad/data/CMIP6/"
era5_dir = "/home/strnad/data/climate_data/Europe"
# %%
reload(at)
reload(cfu)
reload(fut)

country_name = "Germany"

gs = 0.25
cf_dict_tr = {}
tr_historical = [
    ('1980-01-01', '1990-01-01'),
    ('1990-01-01', '2000-01-01'),
    ('2000-01-01', '2010-01-01'),
    ('2010-01-01', '2015-01-01'),
    ('1980-01-01', '2025-01-01'),]

for (start_date, end_date) in tr_historical:
    tr_str = f'{start_date}_{end_date}'

    savepath_dict = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{gs}_{tr_str}.npy'

    if fut.exist_file(savepath_dict):
        cf_dict_cmip = fut.load_np_dict(savepath_dict)
        cf_dict_tr[tr_str] = cf_dict_cmip
    else:
        print(f'CF {tr_str} not found')
# %%
# Plot the weights of the country
reload(cnt)
reload(cfu)
tr_str = '1980-01-01_2025-01-01'
cf_dict = cf_dict_tr[tr_str]

indicator_matrix = cf_dict['onwind']['matrix_xr']
indicator_matrix_off = cf_dict['offwind']['matrix_xr']
indicator_matrix = xr.where(indicator_matrix < indicator_matrix_off,
                            indicator_matrix_off,
                            indicator_matrix)

im = gplt.plot_map(indicator_matrix,
                   vmin=0, vmax=1, plot_borders=True,
                   cmap="Greens", label='Weight',
                   title='Onshore and Offshore Weights')

savepath = f"{config['plot_dir']}/capacities_era5/{country_name}_{gs}_weights_off_on.png"
gplt.save_fig(savepath, fig=im['fig'])

# %%
# plot the corrected data
reload(qm)
im = gplt.create_multi_plot(nrows=1, ncols=len(config['technology']),
                            wspace=0.3)
method = 'quantile_mapping'
# method = 'basic_quantile'
# method = 'normal_mapping'

for idx, (name, cf) in enumerate(cf_dict.items()):
    if name == 'all' or name == 'wind':
        continue
    ts_model, ts_opsd = tu.equalize_time_points(cf['ts_uncorr'], cf['opsd'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        ts_model, ts_opsd
    )
    # QM = qm.BiasCorrection(obs_data=ts_opsd,
    #                        mod_data=ts_model,
    #                        )
    # ts_model_corr = QM.correct(method=method)
    ts_model_corr = cm.adjust(
        method=method,
        obs=ts_opsd,
        simh=ts_model,
        simp=ts_model,
        n_quantiles=1000,
        kind='+',
    )['cf']

    x_lin_reg = np.linspace(0, 1, 100)
    data_lin_reg = slope * x_lin_reg + intercept

    x_unity = np.linspace(0, 1, 100)
    y_unity = x_unity

    gplt.plot_2d(y=[ts_model.data, ts_model_corr, x_lin_reg, x_unity],
                 x=[ts_opsd.data, ts_opsd.data, data_lin_reg, y_unity],
                 ax=im['ax'][idx],
                 ls_arr=['', '', '--', '-'],
                 mk_arr=['x', '+', '', ''],
                 mk_size=3,
                 lw=3,
                 ylim=(0, 1.05),
                 xlim=(0, 1),
                 label_arr=["Uncorr.", "QM-Corrected", "Linear Regression",
                            'Unity'],
                 ylabel="Modelled Cap. Factor [Atlite]",
                 xlabel="Actual Cap. Factor [OPSD]",
                 title=f"{name}",
                 set_grid=True,
                 loc='upper left',)
savepath = f"{config['plot_dir']}/bias_correction/cf_ts_{gs}_{method}_all.png"
gplt.save_fig(savepath, fig=im['fig'])


# %%
reload(gplt)
ncols = len(cf_dict) - 1
im = gplt.create_multi_plot(nrows=1, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.2)
for idx, (sname, source) in enumerate(cf_dict.items()):
    if sname != 'wind':
        cap_fac = source['cf']
        gplt.plot_map(cap_fac,
                      ax=im['ax'][idx],
                      title=sname,
                      vmin=0,
                      label='Capacity Factor [a.u.]',
                      vmax=.2 if sname == 'solar' else 0.4,
                      plot_borders=True)

savepath = f"{config['plot_dir']}/capacities_era5/cf_map_{country_name}_{gs}.png"
gplt.save_fig(savepath, fig=im['fig'])
# %%
# Plot all capacities together
ncols = 3
nrows = len(cf_dict_tr)
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.2)
im_diff = gplt.create_multi_plot(nrows=nrows-1, ncols=ncols,
                                 projection='PlateCarree',
                                 wspace=0.2)
cf_0_dict = cf_dict_tr[list(cf_dict_tr.keys())[0]]
sd, ed = tu.get_time_range(cf_0_dict['wind']['ts'],
                           m=False, d=False, asstr=True)
for tr_idx, (tr_str, cf_dict_cmip) in enumerate(cf_dict_tr.items()):
    sd, ed = tu.get_time_range(cf_dict_cmip['wind']['ts'], asstr=True)
    vertical_title = f'ERA5 \n({sd} - {ed})'

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
                vmin = -0.02 if sname == 'solar' else -0.05
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


savepath = f"{config['plot_dir']}/capacities_era5/cf_types_full_era5_{gs}.png"
gplt.save_fig(savepath, fig=im['fig'])
savepath_diff = f"{config['plot_dir']}/capacities_era5/cf_types_diff_era5_{gs}.png"
gplt.save_fig(savepath_diff, fig=im_diff['fig'])

# %%
# Plot Capacity Factors
reload(gplt)
data_arr = []
timemean = 'dayofyear'
window = 8
ts_type = 'ts'
yearly_ts_all = 0
sd, ed = tu.get_time_range(cf_dict['all']['ts'], asstr=True)
for cap_name, cap in cf_dict.items():
    if cap_name == 'wind':
        continue
    data = cap[ts_type] if cap_name != 'all' else cap['ts']
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
                  #  power_pv.values,
                  title=f"Mean Cap. Factor {sd} - {ed}",
                  label=list(cf_dict.keys()) + ['All'],
                  color_arr=colors,
                  xlabel="Day of Year",
                  set_grid=True,
                  ylim=(0, 1),
                  )
savepath = f"{config['plot_dir']}/capacities_era5/{ts_type}_{gs}_{sd}_{ed}_capacity_factor_per_day.png"
gplt.save_fig(savepath, fig=im['fig'])
# %%
reload(gplt)
subdevide = 4
window = int(365*subdevide)
yearly_ts_all = 0
x_lin_reg = np.linspace(0, 1, 100)
ts_type = 'ts'

colors = ['blue', 'tab:blue', 'orange', 'black']

im = None
for tr_str, cf_dict in cf_dict_tr.items():
    data_arr = []
    trend_arr = []

    for cap_name, cap in cf_dict.items():
        if cap_name == 'wind':
            continue
        da = cap[ts_type] if cap_name != 'all' else cap['ts']
        if window > 1:
            da = tu.rolling_timemean(da, window=window)
        time = da.time
        data_arr.append(da)
        # compute the trend
        num_tps = (time.values - time.values[0]
                   ).astype('timedelta64[h]').astype(int)
        coeffs = np.polyfit(num_tps, da.values, deg=1)
        trend = np.polyval(coeffs, num_tps)
        trend = xr.DataArray(trend, coords={'time': da.time}, dims='time')
        trend_arr.append(trend)

    im = gplt.plot_2d(x=data_arr[0].time,
                      y=data_arr,
                      ax=im['ax'] if im is not None else None,
                      title=f"ERA5 CF",
                      label_arr=list(cf_dict.keys()),
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
                      color_arr=colors,
                      )
    savepath = f"{config['plot_dir']}/capacities_era5/{ts_type}_{gs}_{tr_str}_cf_time_period.png"
    gplt.save_fig(savepath, fig=im['fig'])

# %%
# Compare actual capacity factors with modelled

window = 4
ts_type = 'ts'
for name, cf in cf_dict.items():
    if name == 'all' or name == 'wind':
        continue
    data_arr = []
    ts_opsd = cf['opsd']
    ts_model = cf[ts_type] if name != 'all' else cf['ts']
    ts_opsd, ts_model = tu.equalize_time_points(ts_opsd, ts_model)
    for data in [ts_model, ts_opsd]:
        ts_mean = tu.rolling_timemean(data, window=window)
        # ts_mean = tu.get_time_range_data(ds=ts_mean,
        #                                  time_range=time_range)
        data_arr.append(ts_mean)

    corr = np.corrcoef(data_arr[0].data.flatten(),
                       data_arr[1].data.flatten())[0, 1]
    sd, ed = tu.get_time_range(ts_mean, asstr=True)
    im = gplt.plot_2d(x=data_arr[0].time,
                      y=data_arr,
                      figsize=(10, 3),
                      title=f"Corr CF ({sd} - {ed}) = {corr:.2f}",
                      label_arr=[f"{name} Model", f"{name} OPSD"],
                      color_arr=['blue', 'red', 'black'],
                      xlabel="Day of Year",
                      ylabel="Capacity Factor",
                      ylim=(0, .4) if name == 'solar' else (0, 1),
                      )

    savepath = f"{config['plot_dir']}/opsd_era5/{ts_type}_{gs}_compare_corr_cfs_true_cfs_{name}_{sd}_{ed}_day.png"
    gplt.save_fig(savepath, fig=im['fig'])


# %%
# Compare the distribution of capacity factors for the corrected and uncorrected data
reload(gplt)
im = gplt.create_multi_plot(nrows=1, ncols=3,
                            figsize=(30, 5))
labels = ['Uncorrected', 'QM - Corrected', 'GT (OPSD)']
cf_dict = cf_dict_tr['1980-01-01_2025-01-01']  # Use the latest time range
for idx, (name, cf) in enumerate(cf_dict.items()):
    data_arr = []
    if name == 'all' or name == 'wind':
        continue
    cf_opsd_ts = cf['opsd']
    cf_uncorr_ts, _ = tu.equalize_time_points(cf['ts_uncorr'], cf['opsd'])
    cf_corr_ts, _ = tu.equalize_time_points(cf['ts'], cf['opsd'])

    for i, plot_data in enumerate([cf_uncorr_ts, cf_corr_ts, cf_opsd_ts]):
        gplt.plot_hist(plot_data.data,
                       ax=im['ax'][idx],
                       title=name,
                       color=gplt.colors[i],
                       label=labels[i],
                       nbins=100,
                       lw=2 if i == 2 else 1,
                       alpha=1,
                       xlim=(0, 1) if name != 'solar' else (0, 0.4),
                       xlabel="Capacity Factor",
                       density=True,
                       set_yaxis=True,
                       yscale='log',
                       )


savepath = f"{config['plot_dir']}/opsd_era5/{country_name}_{gs}_compare_cf_distributions.png"
gplt.save_fig(savepath, fig=im['fig'])
# %%
