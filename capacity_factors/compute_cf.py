# %%
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
import yaml
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
# %%
reload(at)
gs = 0.5
country_name = "Germany"
savepath = f'{config['data_dir']}/{country_name}/{config['data'][gs]}'
cutout_germany = at.Cutout(savepath)
reload(cnt)
germany = cnt.get_country('Germany')
germany_shape = cnt.get_country_shape('Germany')
ind_matrix = cnt.cutout_country_cells(cutout_germany,
                                      country_name=country_name,
                                      as_xr=False)

# off shore
reload(cnt)
germany_shape_off = cnt.get_country_shape('Germany', onshore=False)
ind_matrix_off = cnt.cutout_country_cells(cutout_germany,
                                          country_name=country_name,
                                          onshore=False,
                                          as_xr=False)

# %%
# Capacity factors Wind on shore
cf_dict = {'offwind': dict(), 'onwind': dict(), 'solar': dict()}

for wind in ['onwind', 'offwind']:
    turbine = config['technology'][wind]['resource']['turbine']
    cap_wind = cutout_germany.wind(
        turbine=turbine,
        capacity_factor=True)
    power_generation_wind = cutout_germany.wind(
        turbine=turbine, shapes=germany_shape,
        capacity_factor=False)
    power_wind = sput.remove_single_dim(power_generation_wind,)
    matrix = ind_matrix_off if wind == 'offwind' else ind_matrix
    cap_wind_ts = cutout_germany.wind(turbine=turbine,
                                      capacity_factor=False,
                                      layout=cap_wind,
                                      per_unit=True,
                                      matrix=matrix
                                      )
    cap_wind_ts = sput.remove_single_dim(ds=cap_wind_ts)
    cf_dict[wind]['cap'] = cap_wind
    cf_dict[wind]['ts'] = cap_wind_ts
    cf_dict[wind]['power'] = power_wind


# Capacity factors PV
panel = config['technology']['solar']['resource']['panel']
orientation = config['technology']['solar']['resource']['orientation']
cap_pv = cutout_germany.pv(
    panel=panel,
    orientation=orientation,
    capacity_factor=True)

power_generation_pv = cutout_germany.pv(
    shapes=germany_shape,
    panel=panel,
    orientation=orientation,
    capacity_factor=True)
power_pv = sput.remove_single_dim(power_generation_pv)
cap_pv_ts = cutout_germany.pv(panel=panel,
                              orientation=orientation,
                              capacity_factor=False,
                              layout=cap_pv,
                              per_unit=True,
                              matrix=ind_matrix
                              )
cap_pv_ts = sput.remove_single_dim(ds=cap_pv_ts)
cf_dict['solar']['cap'] = cap_pv
cf_dict['solar']['ts'] = cap_pv_ts
cf_dict['solar']['power'] = power_pv

# %%
# OPSD Data
opsd_df = pd.read_csv(config['data']['opsd'])

opsd_times = opsd_df['cet_cest_timestamp']
opsd_onwind_cap = opsd_df['DE_wind_onshore_capacity']
opsd_onwind_gen = opsd_df['DE_wind_onshore_generation_actual']
opsd_offwind_cap = opsd_df['DE_wind_offshore_capacity']
opsd_offwind_gen = opsd_df['DE_wind_offshore_generation_actual']
opsd_pv_cap = opsd_df['DE_solar_capacity']
opsd_pv_gen = opsd_df['DE_solar_generation_actual']

# Transform to xarray
reload(tu)
opsd_onwind_cap_xr = tu.pandas2xr_ts(opsd_onwind_cap, times=opsd_times)
opsd_onwind_gen_xr = tu.pandas2xr_ts(opsd_onwind_gen, times=opsd_times)
opsd_offwind_cap_xr = tu.pandas2xr_ts(opsd_offwind_cap, times=opsd_times)
opsd_offwind_gen_xr = tu.pandas2xr_ts(opsd_offwind_gen, times=opsd_times)
opsd_pv_cap_xr = tu.pandas2xr_ts(opsd_pv_cap, times=opsd_times)
opsd_pv_gen_xr = tu.pandas2xr_ts(opsd_pv_gen, times=opsd_times)

opsd_onwind_cf = opsd_onwind_gen_xr / opsd_onwind_cap_xr
opsd_offwind_cf = sut.normalize(opsd_offwind_gen_xr / opsd_offwind_cap_xr)
opsd_pv_cf = opsd_pv_gen_xr / opsd_pv_cap_xr

hourly_res = 6
opsd_onwind_cf_6h = tu.convert_time_resolution(
    opsd_onwind_cf, keep_time_points=hourly_res,
    freq='h', average=False,
    start_index=1)
opsd_offwind_cf_6h = tu.convert_time_resolution(
    opsd_offwind_cf, keep_time_points=hourly_res,
    freq='h', average=False,
    start_index=1)
opsd_pv_cf_6h = tu.convert_time_resolution(
    opsd_pv_cf, keep_time_points=hourly_res,
    freq='h', average=False,
    start_index=1)

opsd_onwind_cf_6h = tu.remove_nans(opsd_onwind_cf_6h)
opsd_offwind_cf_6h = tu.remove_nans(opsd_offwind_cf_6h)
opsd_pv_cf_6h = tu.remove_nans(opsd_pv_cf_6h)

for name, cf in cf_dict.items():
    cf['opsd'] = opsd_onwind_cf_6h if name == 'onwind' else opsd_offwind_cf_6h if name == 'offwind' else opsd_pv_cf_6h

# %%
# Plot the weights of the country
reload(cnt)
indicator_matrix = cnt.cutout_country_cells(cutout_germany, country_name)
indicator_matrix_off = cnt.cutout_country_cells(
    cutout_germany, country_name, onshore=False)
indicator_matrix = xr.where(indicator_matrix < indicator_matrix_off,
                            indicator_matrix_off,
                            indicator_matrix)

im = gplt.plot_map(indicator_matrix,
                   vmin=0, vmax=1, plot_borders=True,
                   cmap="Greens", label='Weight',
                   title='Onshore and Offshore Weights')

savepath = f"{config['plot_dir']}/{country_name}_{gs}_weights_off_on.png"
gplt.save_fig(savepath, fig=im['fig'])


# %%
# plot the corrected data
im = gplt.create_multi_plot(nrows=1, ncols=len(cf_dict),
                            wspace=0.3)
method = 'quantile_mapping'
for idx, (name, cf) in enumerate(cf_dict.items()):
    ts_model, ts_opsd = tu.equalize_time_points(cf['ts'], cf['opsd'])

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        ts_model, ts_opsd
    )
    QM = qm.BiasCorrection(obs_data=ts_opsd,
                           mod_data=ts_model)
    ts_model_corr = QM.correct(method=method)

    x_lin_reg = np.linspace(0, 1, 100)
    data_lin_reg = slope * x_lin_reg + intercept

    x_unity = np.linspace(0, 1, 100)
    y_unity = x_unity

    gplt.plot_2d(y=[ts_model.data, ts_model_corr.data, x_lin_reg, x_unity],
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
savepath = f"{config['plot_dir']}/{country_name}_{gs}_{method}_all.png"
gplt.save_fig(savepath, fig=im['fig'])
# %%
# Now correct the full time series
reload(qm)
for idx, (name, cf) in enumerate(cf_dict.items()):
    ts_model = cf['ts']
    ts_opsd = cf['opsd']

    QM = qm.BiasCorrection(obs_data=ts_opsd,
                           mod_data=ts_model)
    ts_model_corr = QM.correct(method=method)
    cf_dict[name]['ts'] = ts_model_corr
# %%
data_arr = []
weight_arr = []
ts_all = 0
for cap_name, cap in cf_dict.items():
    data = cap['ts']
    weight = config['technology'][cap_name]['weight']
    ts_all += data * weight
    weight_arr.append(weight)

ts_all = ts_all / np.sum(weight_arr)

# %%
# save the time series
reload(fut)
savepath_dict = f'{config['data_dir']}/{country_name}/cf_dict_{gs}.npy'
fut.save_np_dict(cf_dict, savepath_dict)

# %%
reload(gplt)
ncols = len(cf_dict)
im = gplt.create_multi_plot(nrows=1, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.2)
for idx, (sname, source) in enumerate(cf_dict.items()):
    cap_fac = source['cap']
    gplt.plot_map(cap_fac,
                  ax=im['ax'][idx],
                  title=sname,
                  # vmin=0, #vmax=.3,
                  plot_borders=True)

savepath = f"{config['plot_dir']}/{country_name}_cap_facts_{gs}.png"
gplt.save_fig(savepath, fig=im['fig'])


# %%
# Plot Power Generations
reload(gplt)
reload(tu)
time_range = ['2018-01-01', '2018-12-31']

data_arr = []
timemean = 'dayofyear'

for cap_name, cap in cf_dict.items():
    data = cap['power']
    ts_mean = tu.rolling_timemean(data, window=8)
    yearly_ts = tu.compute_timemean(ts_mean,
                                    timemean=timemean,
                                    groupby=True,
                                    reset_time=True,)
    data_arr.append(yearly_ts)

data_arr.append(sum(data_arr))

sd, ed = tu.get_time_range(power_wind, asstr=True)
im = gplt.plot_2d(x=data_arr[0].time,
                  y=data_arr,
                  #  power_pv.values,
                  title=f"Mean Renewable Power Generation {sd} - {ed}",
                  label_arr=list(cf_dict.keys()) + ['Sum'],
                  xlabel="Day of Year",
                  ylabel="Power Generation [GWh]",
                  #   ylim=(0, 3.5),
                  )
savepath = f"{config['plot_dir']}/{country_name}_{gs}_power_generation_per_day.png"
gplt.save_fig(savepath, fig=im['fig'])


# %%
# Plot Capacity Factors
data_arr = []
timemean = 'dayofyear'
window = 8
yearly_ts_all = 0
for cap_name, cap in cf_dict.items():
    data = cap['ts']
    ts_mean = tu.rolling_timemean(data, window=window)
    weight = config['technology'][cap_name]['weight']
    yearly_ts = tu.compute_timemean(ts_mean,
                                    timemean=timemean,
                                    groupby=True,
                                    reset_time=True,)
    yearly_ts_all += yearly_ts * weight

    data_arr.append(yearly_ts)

data_arr.append(yearly_ts_all)

sd, ed = tu.get_time_range(data, asstr=True)
im = gplt.plot_2d(x=data_arr[0].time,
                  y=data_arr,
                  #  power_pv.values,
                  title=f"Mean Cap. Factor {sd} - {ed}",
                  label_arr=list(cf_dict.keys())+ ['All'],
                  color_arr=['blue', 'tab:blue', 'orange', 'black'],
                  xlabel="Day of Year",
                  set_grid=True,
                  ylim=(0, 1),
                  )
savepath = f"{config['plot_dir']}/{country_name}_{gs}_capacity_factor_per_day.png"
gplt.save_fig(savepath, fig=im['fig'])

# %%
# Compare actual capacity factors with modelled
time_range = ['2017-01-01', '2017-12-31']

window = 4
for name, cf in cf_dict.items():
    data_arr = []
    for data in [cf['ts'], cf['opsd']]:
        data = tu.rolling_timemean(data, window=window)
        ts_mean = tu.get_time_range_data(ds=data,
                                         time_range=time_range)
        data_arr.append(ts_mean)

    sd, ed = tu.get_time_range(ts_mean, asstr=True)
    im = gplt.plot_2d(x=data_arr[0].time,
                      y=data_arr,
                      figsize=(10, 3),
                      #  power_pv.values,
                      title=f"Actual Cap. Factor {sd} - {ed}",
                      label_arr=[f"{name} Model", f"{name} OPSD"],
                      color_arr=['blue', 'red', 'black'],
                      xlabel="Day of Year",
                      ylim=(0, 1.2),
                      )

    savepath = f"{config['plot_dir']}/{country_name}_{gs}_compare_true_capacity_factor_{name}_{sd}_{ed}_day.png"
    gplt.save_fig(savepath, fig=im['fig'])


# %%
# Compute the Capacity factors per grid cell
reload(tu)
reload(gplt)

turbine = "Vestas_V112_3MW"
cap_wind_grid = cutout_germany.wind(
    turbine=turbine,
    capacity_factor=True,
    capacity_factor_timeseries=True)  # Computes the capacity factor per grid cell


cap_pv_grid = cutout_germany.pv(
    panel=panel,
    orientation=orientation,
    capacity_factor=True,
    capacity_factor_timeseries=True)

# %%
#
savepath_wind = f"{config['data_dir']}/{country_name}/cap_wind_{gs}.nc"
savepath_pv = f"{config['data_dir']}/{country_name}/cap_pv_{gs}.nc"

fut.save_ds(ds=cap_wind_grid, filepath=savepath_wind)
fut.save_ds(ds=cap_pv_grid, filepath=savepath_pv)

