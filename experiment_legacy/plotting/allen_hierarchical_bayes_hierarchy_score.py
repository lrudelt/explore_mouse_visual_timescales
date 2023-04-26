#%%
import argparse
from sys import exit, path
import os
import yaml
with open('dirs.yaml', 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import patsy as pt
import scipy.stats as st
import arviz as az
import theano

path.insert(1, "../../allen_src/")
import analysis_utils as utl
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors
import seaborn as sns
import importlib
importlib.reload(utl);

#%%
args = argparse.Namespace()
defined_measures = ["tau_C",
                    "tau_R",
                    "R_tot"]
defined_stimuli = ["movie",
                   "spontaneous"]

#%%
parser = argparse.ArgumentParser()
parser.add_argument('measure', type=str, help=f'one of {defined_measures}')
parser.add_argument('stimulus', type=str, help=f'one of {defined_stimuli}, default: movie', nargs='?', default="movie")
parser.add_argument('--bo', dest='allen_bo', action='store_true',
                    help=f'use brain observatory data set for movie stimulus')
args = parser.parse_args()
if not args.measure in defined_measures:
    parser.print_help()
    exit()
if not args.stimulus in defined_stimuli:
    parser.print_help()
    exit()
if args.allen_bo:
    if not args.stimulus == "movie":
        parser.print_help()
        exit()
args = parser.parse_args()

# analysis parameters
#%%
# args.measure = 'R_tot'
# args.stimulus = 'spontaneous'
# args.allen_bo = False
# __file__ = 'allen_hierarchical_bayes_hierarchy_score.py'

seed = 12347

center = 'median' # measure of central tendency
T_measure = 'tau_R'
# T_measure = 'log_tau_R'
R_measure = 'R_tot'
# C_measure = 'mre_tau'
C_measure = 'tau_two_timescales'

if args.allen_bo:
    analysis = 'allen_bo'
else:
    analysis = 'allen_fc'

if args.allen_bo:
    stimuli = ['natural_movie_three']
elif args.stimulus == "movie":
    stimuli = ['natural_movie_one_more_repeats']
elif args.stimulus == "spontaneous":
    stimuli = ['spontaneous']

T0 = 0.03 # 30 ms
#selected_structures = 'cortex+thalamus'
selected_structures = 'cortex'

plot_stars = False # plot stars or p-values for tests

# setup analysis
#%%
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])
plot_settings['imgdir'] = '../img'

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
# mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_10000_two_timescales.csv".format(stats_dir, analysis)

structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP']
# structures = ['VISp','VISl','VISrl','VISal','VISpm','VISam']
structures_map = utl.get_structures_map()
structures_cortex = [structure for structure in structures
                     if structures_map[structure]['parent_structure'] == 'cortex']

data_directory = dir_settings['allen_data_dir']
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

### import data

analysis_metrics = utl.get_analysis_metrics(cache,
                                            analysis)
data = utl.get_analysis_data(csv_file_name, analysis,
                             analysis_metrics=analysis_metrics,
                             mre_stats_file_name=mre_stats_file_name)

# make sure data as expected
if (selected_structures == 'cortex+thalamus') :
    _structures = structures
elif (selected_structures == 'cortex') :
    _structures = structures_cortex
else:
    print('unknown structures selected')
    exit()

data = data[utl.df_filter(data, structures=_structures, stimuli=stimuli, T0=T0, tmin=T0)]
try:
    num_neurons = utl.get_expected_number_of_neurons(analysis,
                                                     selected_structures,
                                                     stimuli)
    assert np.isclose(len(data), num_neurons, atol=100)
except:
    print(f'number of neurons is {len(data)}, expected {num_neurons}')
    exit()

if args.measure == "tau_C":
    measure = C_measure
    measure_name = r'intrinsic timescale'# $τ_{\mathregular{C}}$ (ms)'
    measure_name_short = '$τ_{\mathregular{C}}$ (ms)'
elif args.measure == "tau_R":
    measure = T_measure
    measure_name = 'information timescale'# $τ_R$ (ms)'
    measure_name_short = '$τ_R$'
elif args.measure == "R_tot":
    measure = R_measure
    measure_name = r'predictability'# $R_{\mathregular{tot}}$'
    measure_name_short = r'$R_{\mathregular{tot}}$'

selection = utl.get_data_filter(data, measure)
data = data[selection]

area_dict = dict(zip(_structures, range(len(_structures))))

# MODEL 2: LINEAR HIERARCHY MODEL
#%%
model_name = "hierarchy_score"

## set up model

ft_endog = measure
# fts_num = ['log_fr', 'hierarchy_score'] # 'firing_rate'
fts_num = ['log_fr'] # 'firing_rate'
fts_cat = ['sign_rf', 'session_id']


# mean-centered & standardized according to Gelman: 1 / (2 * sd)
# Divide by 2 standard deviations in order to put the variance of a
# normally distributed variable nearer to the variance range of a
# binary variable. See
# http://www.stat.columbia.edu/~gelman/research/published/standardizing7.pdf
# for more info.

# if measure == R_measure or measure == C_measure:
ft_endog_data = np.log(data[ft_endog])
# elif measure == T_measure:
#     ft_endog_data = data[ft_endog]

ft_endog_mean = ft_endog_data[ft_endog_data > -np.inf].mean(0)
ft_endog_sd = ft_endog_data[ft_endog_data > -np.inf].std(0)

_data = pd.concat(((ft_endog_data - ft_endog_mean) / ft_endog_sd,
                   data[fts_cat],
                   ((data[fts_num] - data[fts_num].mean(0)) / (2 * data[fts_num].std(0))),
                   ((data['hierarchy_score'] - data['hierarchy_score'].min(0)) / (2 * data['hierarchy_score'].std(0)))),1)
_data = _data[_data[ft_endog] > -np.inf]

for var in _data.columns.values:
    _data = _data[~_data[var].isna()]

#fml_all = '{} ~ '.format(ft_endog) + ' + '.join(fts_num + fts_cat)

#(mx_en, mx_ex) = pt.dmatrices(fml_all, _data, return_type='dataframe')#, NA_action='raise')
# print(mx_ex.describe())
# print(mx_ex.columns.values)

for var in _data.columns.values:
    _data[var] = _data[var].astype(theano.config.floatX)

session_idxs, sessions = pd.factorize(_data.session_id)
coords = {
    "session": sessions,
    "obs_id": np.arange(len(session_idxs))
}

with pm.Model(coords=coords) as model:
    session_idx = pm.Data("session_idx", session_idxs, dims="obs_id")
    hierarchy_score = pm.Data("hierarchy_score", _data.hierarchy_score.values, dims="obs_id")
    # hyperpriors
    mu_intercept = pm.Normal('mu_intercept', mu=0., sigma=1)
    sigma_intercept = pm.HalfCauchy('sigma_intercept', beta=1)
    # mu_rf = pm.Normal('mu_rf', mu=0., sigma=1)
    # sigma_rf = pm.HalfCauchy('sigma_rf', beta=1)
    # mu_log_fr = pm.Normal('mu_log_fr', mu=0., sigma=1)
    # sigma_log_fr = pm.HalfCauchy('sigma_log_fr', beta=1)
    mu_hierarchy_slope = pm.Normal('mu_hierarchy_slope', mu=0., sigma=1)
    sigma_hierarchy_slope = pm.HalfCauchy('sigma_hierarchy_slope', beta=1)

    session_intercept = pm.Normal('session_intercept', mu=0.0, sigma=1.0, dims="session")
    # b0 = pm.Normal('b0_intercept', mu=mu_intercept, sigma=sigma_intercept, dims="session")
    session_hierarchy_slope = pm.Normal('session_hierarchy_slope', mu=0.0, sigma=1.0, dims="session")
    # b1 = pm.Normal('hierarchy_score', mu=mu_area, sigma=sigma_area, dims="session")
    b_sign_rf = pm.Normal('b_sign_rf', mu=0, sigma=1)
    # b1 = pm.Normal('sign_rf', mu=mu_rf, sigma=sigma_rf, dims="session")
    b_log_fr = pm.Normal('b_log_fr', mu=0, sigma=1)
    # b3 = pm.Normal('log_fr', mu=mu_log_fr, sigma=sigma_log_fr, dims="session")

    # define linear model
    yest = ( mu_intercept + session_intercept[session_idx] * sigma_intercept +
             (mu_hierarchy_slope + session_hierarchy_slope[session_idx]*sigma_hierarchy_slope) * _data.hierarchy_score.values +
             b_sign_rf * _data.sign_rf.values +
             b_log_fr * _data.log_fr
    )

    # define normal likelihood with halfnormal error
    epsilon = pm.HalfCauchy('epsilon', 10)
    if args.measure == "R_tot":
        likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon, observed=_data[ft_endog], dims="obs_id")
    else:
        alpha = pm.Normal('alpha', mu = 0., sigma=1)
        likelihood = pm.SkewNormal('likelihood', mu=yest, sigma=epsilon, alpha=alpha, observed=_data[ft_endog], dims="obs_id")

# Transfer variables back from log to linear space, and multiply by standard deviation of log because of transform of the data before modelling
f_transf = lambda x: np.exp(x * ft_endog_sd)
f_transf_log = lambda x: x * ft_endog_sd
f_transf_log_int = lambda x: (x * ft_endog_sd + ft_endog_mean)

if measure == R_measure:
    f_transf_int = lambda x: np.exp(x * ft_endog_sd + ft_endog_mean)
    f_transf_int_inv = lambda x: (np.log(x) - ft_endog_mean) / ft_endog_sd

elif measure == T_measure or measure == C_measure:
    f_transf_int = lambda x: np.exp(x * ft_endog_sd + ft_endog_mean) * 1000
    f_transf_int_inv = lambda x: (np.log(np.divide(x, 1000)) - ft_endog_mean) / ft_endog_sd

with model:
    eff_session_hierarchy_slope = pm.Deterministic("eff_session_hierarchy_slope", mu_hierarchy_slope + session_hierarchy_slope * sigma_hierarchy_slope)
    eff_session_intercept = pm.Deterministic("eff_session_intercept", mu_intercept + session_intercept * sigma_intercept)
    linear_fit = pm.Deterministic("linear_fit", f_transf_int(mu_intercept + mu_hierarchy_slope * hierarchy_score), dims="obs_id")

pm.model_to_graphviz(model)

 ## model visualization
#
# fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
# az.plot_dist(f_transf_int(mu_intercept.random(size=5000)))
# if measure == R_measure:
#     ax0.set_xlabel('$θ_0$')
# elif measure == T_measure or measure == C_measure:
#     ax0.set_xlabel('$θ_0$ (ms)')
# ax0.set_ylabel('$p(θ_0)$')
# ax0.set_title('')
# utl.make_plot_pretty(ax0)
#
# utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_intercept", measure=args.measure)
#
#
# fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
# az.plot_dist(b2.random(size=5000))
# ax0.set_xlabel('log($θ_1$)')
# ax0.set_ylabel('$p($log($θ_1))$')
# ax0.set_title('')
# utl.make_plot_pretty(ax0)
#
# ax0.text(1.1, 0.28,
#          r'$\mathcal{N}(\mu=0,\sigma=1)$', fontsize=8)
#
# utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_coeff", measure=args.measure)

## prior predictive checks

with model:
    prior_pred_samples = pm.sample_prior_predictive(samples=500, random_seed=seed)
#
# fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
# ax0.set_title('')
# utl.make_plot_pretty(ax0)
#
# # np.unique((data['hierarchy_score'] - data['hierarchy_score'].mean(0)) / (2 * data['hierarchy_score'].std(0)))
# #_y = f_transf_int(prior_pred_samples['b0_intercept']
# #                  + (-0.71806953)*prior_pred_samples['hierarchy_score']
# #                  + prior_pred_samples['sign_rf[T.True]'])
# _y = f_transf_int(prior_pred_samples['mu_intercept']
#                   + (-0.71806953)*prior_pred_samples['mu_hierarchy_slope']
#                   + prior_pred_samples['b_sign_rf'])
#
# ax0.hist(_y[_y<10000], bins=30,
#          density=True, color = 'k', lw = 0, zorder = -1, rwidth=0.9)
# if measure == R_measure:
#     ax0.set_xlabel(r'$θ_0 \cdot θ_1^{rf} \cdot θ_2^{V1}$')
# elif measure == T_measure or measure == C_measure:
#     ax0.set_xlabel(r'$θ_0 \cdot θ_1^{rf} \cdot θ_2^{V1}$ (ms)')
#
# ax0.set_ylabel('ρ')
# if measure == T_measure:
#     ax0.set_xlim([-0.01,1010])
# elif measure == C_measure:
#     ax0.set_xlim([-0.01,10001])
# #elif measure == R_measure:
# #    ax0.set_xlim([-0.01, 1.01])
#
# utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_predictive", measure=args.measure)

## posterior sampling
#%%
with model:
    ## take samples
    trc = pm.sample(3000, tune=2000, random_seed=seed, target_accept = 0.95)

    print(pm.summary(trc).round(2))

#%%
# az.plot_trace(trc, compact = True, chain_prop ={"ls": "-"})

## posterior visualization (non-hierarchical parameters)
#%%
with model:
    if args.measure == "R_tot":
        n_axes = 3
        fig_height = 4.5
    else:
        n_axes = 4
        fig_height = 6
    fig1, axes = plt.subplots(n_axes, 1, figsize=(plot_settings["panel_width"], fig_height))
    fig1.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=1)
    axes[0].axvline(x=1, ls = "--", lw =2, color = "0.0")    
    axes[1].axvline(x=1, ls = "--", lw =2, color = "0.0")   
    # utl.plot_posterior(trc[-250:], var_names=['mu_intercept'],
    #                    #coords=0,
    #                    ax=axes[0], point_estimate='median',
    #                    hdi_prob=0.95,
    #                    transform=f_transf_int)

    utl.plot_posterior(trc[-1000:], var_names=[#'b0_intercept',
                                              'b_sign_rf',
                                              #'stimulus[T.spontaneous]',
                                              #'sign_rf[T.True]:stimulus[T.spontaneous]',
                                              # 'firing_rate',
                                              'b_log_fr'],
                       #coords=0,
                       ax=axes[:2], point_estimate='median',
                       hdi_prob=0.95,
                       transform=f_transf)
    utl.plot_posterior(trc[-1000:], var_names=['epsilon'],
                       ax=axes[2], point_estimate='median',
                       hdi_prob=0.95,
                       transform=f_transf_log)
    if not args.measure == "R_tot":
        utl.plot_posterior(trc[-1000:], var_names=['alpha'],
                           ax=axes[3], point_estimate='median',
                           hdi_prob=0.95)
    vars = [r'$\mathrm{θ_{rf}}$', r'$θ_{\log \nu}$', 'ε', r'$\alpha$']
    var_names = [r'$\exp(\mathrm{θ_{rf}})$ (responsiveness)', r'$\exp(θ_{\log \nu})$ (log fir. rate)', 'ε (scale)', r'$\alpha$ (shape)']
    for ax, var, var_name in zip(axes, vars[:n_axes], var_names[:n_axes]):
#'$θ_0$ (no rf, fir. rate={:.2f}Hz)'.format(10**np.mean(data['log_fr'])),
        #ax.set_xlabel(var)
        ax.set_ylabel('p({} | E)'.format(var))
        #ax.set_title(var_name)
        ax.set_xlabel(var_name)
        ax.set_title('')
        utl.make_plot_pretty(ax)

    axes[0].set_title(measure_name, ha='center')
    #axes[0].set_xticks([0.06, 0.065])
    #axes[1].set_xticks([-0.075, -0.08, -0.004])
    #axes[2].set_xticks([-0.012, -0.008, -0.004])
    #axes[3].set_xticks([-0.015, -0.01, -0.005])
    #axes[4].set_xticks([0.45, 0.5])
    #axes[5].set_xticks([0.048, 0.049, 0.05])

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior", allen_bo=args.allen_bo, stimulus=args.stimulus, measure=args.measure)

# with model:
#     fig0, ax0 = plt.subplots(1, 1, figsize=(plot_settings["panel_width"], 2))
#     fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=1)
#
#     utl.plot_posterior(trc[-250:], var_names=['mu_area'],
#                        ax=ax0, point_estimate='median',
#                        hdi_prob=0.95,
#                        transform=f_transf)
#
#     ax0.set_ylabel('p({$θ_3$} | E)')
#     ax0.set_xlabel('$θ_3$ (hier. score)')
#     ax0.set_title('')
#     utl.make_plot_pretty(ax0)
#
#     axes[0].set_title(measure_name, ha='center')
#
# utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior_hierarchy_score", measure=args.measure)

# posterior visualization (hierarchical parameters)

idata = az.from_pymc3(
    trace=trc,
    prior=prior_pred_samples,
    model=model,
)

post = idata.posterior.assign_coords(session_idx=idata.constant_data.session_idx)

# slope

fig0 = plt.figure(figsize=(plot_settings["panel_width"]*1.7, 3))
# fig0.suptitle(measure_name, ha = "center")
# fig0.suptitle("hierarchy score slope", ha = "center")

ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)
fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=0.6, wspace=0.4)
# if args.measure == "tau_C":
#     ax1.set_xlim([-0.05,0.22])
#     ax3.set_xlim([-0.24,0.44])
# if args.measure == "tau_R":
#     ax1.set_xlim([-0.02,0.14])
#     ax3.set_xlim([-0.12,0.27])
# if args.measure == "R_tot":
#     ax1.set_xlim([-0.18,0.06])
#     ax3.set_xlim([-0.31,0.21])

with model:
    utl.plot_posterior(post["mu_hierarchy_slope"].values[:,-1000:],
                       ax=ax1, point_estimate='median',
                       hdi_prob=0.95,
                       transform = f_transf_log)
    ax1.set_xlabel(r'mean slope $\mu_{\theta_{\mathrm{hs}}}$')
    ax1.set_ylabel(r'$p\left(\mu_{\theta_{\mathrm{hs}}} | E\right)$')
    utl.plot_posterior(post["sigma_hierarchy_slope"].values[:,-1000:],
                       ax=ax2, point_estimate='median',
                       hdi_prob=0.95,
                       transform = f_transf_log)
    ax2.set_xlabel(r'std slope $\sigma_{\theta_{\mathrm{hs}}}$')
    ax2.set_ylabel(r'$p\left(\sigma_{\theta_{\mathrm{hs}}} | E\right)$')


    for session_idx in range(len(sessions)):
        utl.plot_posterior(post["eff_session_hierarchy_slope"].values[:,-1000:,session_idx],
                       ax=ax3,
                       hdi = False,
                       color = sns.color_palette()[session_idx%10],
                       transform = f_transf_log)
    ax3.set_xlabel(r'hierarchy score slope $\theta_{\mathrm{hs}}$')
    ax3.set_ylabel(r'$p\left(\theta_{\mathrm{hs}} | E\right)$')


for ax in [ax1,ax2,ax3]:
    utl.make_plot_pretty(ax)

ax1.axvline(x=0, ls = "--", lw =2, color = "0.0")    
ax3.axvline(x=0, ls = "--", lw =2, color = "0.0")    

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior_slope", allen_bo=args.allen_bo, stimulus=args.stimulus, measure=args.measure)

# intercept

fig0 = plt.figure(figsize=(plot_settings["panel_width"]*1.6, 3))
# fig0.suptitle(measure_name, ha = "center")
# fig0.suptitle("intercept", ha = "center")

ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)
fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=0.6, wspace=0.3)


with model:
    utl.plot_posterior(post["mu_intercept"].values[:,-1000:],
                       ax=ax1, point_estimate='median',
                       hdi_prob=0.95,
                       transform = f_transf_log_int)

    ax1.set_xlabel(r'mean intercept $\mu_{\theta_0}$')
    ax1.set_ylabel(r'$p\left(\mu_{\theta_0} | E\right)$')
    utl.plot_posterior(post["sigma_intercept"].values[:,-1000:],
                       ax=ax2, point_estimate='median',
                       hdi_prob=0.95,
                       transform = f_transf_log)
    ax2.set_xlabel(r'std intercept $\sigma_{\theta_0}$')
    ax2.set_ylabel(r'$p\left(\sigma_{\theta_0} | E\right)$')

    for session_idx in range(len(sessions)):
        utl.plot_posterior(post["eff_session_intercept"].values[:,-1000:,session_idx],
                       ax=ax3,
                       transform = f_transf_int,
                       hdi = False,
                       color = sns.color_palette()[session_idx%10])
    if args.measure == "R_tot":
        ax3.set_xlabel(r'intercept $\exp(\theta_0)$')
    else:
        ax3.set_xlabel(r'intercept $\exp(\theta_0)$ (ms)')

    ax3.set_ylabel(r'$p\left(\theta_0 | E\right)$')
    # ax3.set_xlim([-0.55,0.22])

for ax in [ax1,ax2,ax3]:
    utl.make_plot_pretty(ax)

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior_intercept",  allen_bo=args.allen_bo, stimulus=args.stimulus, measure=args.measure)

# Hierarchy plot 
#%%

# fig0, ax = plt.subplots()
# fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)

# if args.allen_bo:
#     if args.measure == "tau_C":
#         lims = [0.29, 0.76]
#         y_pos = 0.31
#         ax.set_title('intrinsic timescale')
#         measure = C_measure
#     elif args.measure == "tau_R":
#         lims = [0.029, 0.086]
#         y_pos = 0.031
#         ax.set_title('information timescale')
#         measure = T_measure
#     elif args.measure == "R_tot":
#         lims = [0.059, 0.093]
#         y_pos = 0.061
#         ax.set_title('predictable information')
#         measure = R_measure
# elif args.stimulus == "movie":
#     if args.measure == "tau_C":
#         lims = [0.17, 0.52]
#         y_pos = 0.18
#         ax.set_title('intrinsic timescale')
#         measure = C_measure
#     elif args.measure == "tau_R":
#         lims = [0.026, 0.076]
#         y_pos = 0.031
#         ax.set_title('information timescale')
#         measure = T_measure
#     elif args.measure == "R_tot":
#         lims = [0.064, 0.094]
#         y_pos = 0.066
#         ax.set_title('predictable information')
#         measure = R_measure
# elif args.stimulus == "spontaneous":
#     if args.measure == "tau_C":
#         lims = [0.17, 0.58]
#         y_pos = 0.21
#         ax.set_title('intrinsic timescale')
#         measure = C_measure
#     elif args.measure == "tau_R":
#         lims = [0.017, 0.076]
#         y_pos = 0.021
#         ax.set_title('information timescale')
#         measure = T_measure
#     elif args.measure == "R_tot":
#         lims = [0.049, 0.074]
#         y_pos = 0.051
#         ax.set_title('predictable information')
#         measure = R_measure

# x_pos = -0.2

# # elif stimulus == 'spontaneous':
# #     lim_R = [0.049, 0.074]
# #     lim_T = [0.017, 0.076]
# #     lim_C = [0.17, 0.76]

# #     x_pos_text = [-0.2, -0.2, -0.2]
# #     y_pos_text = [0.21, 0.021, 0.051]

# x = []
# y = []
# y_err = []

# for structure in _structures:
#     _x = structures_map[structure]['hierarchy_score']
#     _y = utl.get_center(data[utl.df_filter(data,
#                                            structures=structure)][measure].values,
#                         center)
#     _y_err = utl.get_sd(data[utl.df_filter(data,
#     structures=structure)][measure].values,
#     center)
#     if measure == 'median_ISI':
#         _y = _y * 1000 # from s to ms
#         _y_err = _y_err[0] * 1000 # from s to ms

#     ax.plot([_x, _x],
#             [_y - _y_err, _y + _y_err],
#             lw=2,
#             color=structures_map[structure]['color'],
#     )
#     ax.plot(_x,
#             _y,
#             'o',
#             color="white",
#             mec=structures_map[structure]['color']
#     )


#     if structure in structures_cortex:
#         x += [_x]
#         y += [_y]
#         y_err += [_y_err]

# ax.set_xticks([-0.5, -0.25, 0, 0.25])
# utl.make_plot_pretty(ax)
# ax.set_ylim(lims)

# slope,intercept,r,p,std = st.linregress(x, y)
# x2 = np.linspace(min(x), max(x), 10)

# ax.plot(x2, x2*slope+intercept, '--k', alpha=0.5)

# r_s, p_s = st.spearmanr(x, y)
# r_p, p_p = st.pearsonr(x, y)

# text = ""
# if p_p < 0.0005:
#     text += '$r_P$ = {:.2f}; '.format(r_p) + '$P_P < 10^{-3}$\n'
# else:
#     text += '$r_P$ = {:.2f}; $P_P$ = {:.3f}\n'.format(r_p, p_p)
# if p_s < 0.0005:
#     text += '$r_S$ = {:.2f}; '.format(r_s) + '$P_S$ < 10^{-3}'
# else:
#     text += '$r_S$ = {:.2f}; $P_S$ = {:.3f}'.format(r_s, p_s)

# # + str(np.around(pow(r_p,1),2)) + '' + str(np.around(p_p,6)) + '\n' + \
# #         '$r_S$ = ' + str(np.around(pow(r_s,1),2)) + '; $P_S$ = ' + str(np.around(p_s,6))
# ax.text(x_pos, y_pos, text, fontsize=8)

# # plot Bayesian mean and HDI for slope
# hierarchy_score = idata.constant_data.hierarchy_score
# post = idata.posterior.assign_coords(hierarchy_score=hierarchy_score)
# avg_linear_fit = post["linear_fit"].mean(dim=("chain", "draw")).sortby("hierarchy_score")

# ax.plot(avg_linear_fit.hierarchy_score, avg_linear_fit, "k--", alpha=0.6, label="Mean linear fit")
# az.plot_hdi(
#     hierarchy_score,
#     post["linear_fit"],
#     fill_kwargs={"alpha": 0.1, "color": "k", "label": "Mean linear fit HPD"},
#     ax=ax,
# )

# utl.format_ax(ax, measure, 'y', set_ticks=False, tight_layout=True)

# ax.set_xlabel('hierarchy score')

# # legend_elements = [Line2D([0], [0], marker='s', color=structures_map[structure]['color'],
# #                           label=structures_map[structure]['name'],
# #                           markerfacecolor=structures_map[structure]['color'],
# #                           ms=10,
# #                           linewidth=0)
# #                    for structure in _structures]
# # legend = axes[2].legend(handles=legend_elements,
# #                         fancybox=False,
# #                         loc="lower right",
# #                         bbox_to_anchor=(1.55, 0.10))
# # frame = legend.get_frame()
# # frame.set_facecolor('0.9')
# # frame.set_edgecolor('0.9')

# utl.save_plot(plot_settings, f"{__file__[:-3]}", allen_bo=args.allen_bo, stimulus=args.stimulus, measure=args.measure)


#
# post_pred = idata.posterior_predictive.assign_coords(session_idx=idata.constant_data.session_idx)
# post = idata.posterior.assign_coords(session_idx=idata.constant_data.session_idx)
#
# ## correlation with hierarchy score
#
# hs_mean = data['hierarchy_score'].mean(0)
# hs_2sd = 2 * data['hierarchy_score'].std(0)
#
# measure_per_structure = {}
# measure_intercept = post['b0_intercept'].values[:,-250:,:].flatten()
# for structure in structures_cortex:
#     hs = (structures_map[structure]['hierarchy_score'] - hs_mean) / hs_2sd
#
#     measure_per_structure[structure] \
#     = f_transf_int(measure_intercept
#                    + hs * post['hierarchy_score'].values[:,-250:,:].flatten())
#
# fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
#
# x = []
# y = []
#
# for structure in structures:
#     # _x = structures_map[structure]['hierarchy_score']
#     # # _y = measure_per_structure[structure]
#     # _y = np.median(measure_per_structure[structure])
#
#     if structure in structures_cortex:
#         x += [structures_map[structure]['hierarchy_score']]
#         y += [np.median(measure_per_structure[structure])]
#
# utl.plot_hdi(measure_per_structure, structures_map=structures_map, ax=ax0)
#
# ax0.set_ylabel(measure_name_short)
# ax0.set_xlabel('hierarchy score')
#
#
# if measure == T_measure:
#     lims = [29, 64]
#     xpos = -0.2
#     ypos = 31
# elif measure == C_measure:
#     lims = [190, 620]
#     xpos = -0.2
#     ypos = 220
# elif measure == R_measure:
#     lims = [0.066, 0.093]
#     xpos = -0.2
#     ypos = 0.067
#
# ax0.set_title(measure_name)
# ax0.set_ylim(lims)
# utl.make_plot_pretty(ax0)
# ax0.set_xticks([-0.5,-0.25,0,0.25])
#
# # legend
# # legend_elements = [Line2D([0], [0], marker='s', color=structures_map[structure]['color'],
# #                           label=structures_map[structure]['name'],
# #                           markerfacecolor=structures_map[structure]['color'],
# #                           ms=10,
# #                           linewidth=0)
# #                    for structure in structures]
# # legend = ax0.legend(handles=legend_elements,
# #                     fancybox=False,
# #                     loc="lower right",
# #                     bbox_to_anchor=(1.40, 0.1))
# # frame = legend.get_frame()
# # frame.set_facecolor('0.9')
# # frame.set_edgecolor('0.9')
#
# utl.save_plot(plot_settings, f"{__file__[:-3]}_hierarchy", measure=args.measure)

# STRAIGHT MODEL COMPARISON
# with pm.Model(coords=coords) as session_specific_variance_model:
#     session_idx = pm.Data("session_idx", session_idxs, dims="obs_id")
#
#     # hyperpriors
#     mu_intercept = pm.Normal('mu_intercept', mu=0., sigma=1)
#     sigma_intercept = pm.HalfCauchy('sigma_intercept', beta=1)
#     # mu_rf = pm.Normal('mu_rf', mu=0., sigma=1)
#     # sigma_rf = pm.HalfCauchy('sigma_rf', beta=1)
#     # mu_log_fr = pm.Normal('mu_log_fr', mu=0., sigma=1)
#     # sigma_log_fr = pm.HalfCauchy('sigma_log_fr', beta=1)
#     mu_hierarchy_slope = pm.Normal('mu_hierarchy_slope', mu=0., sigma=1)
#     sigma_hierarchy_slope = pm.HalfCauchy('sigma_hierarchy_slope', beta=1)
#
#     session_intercept = pm.Normal('session_intercept', mu=0.0, sigma=1.0, dims="session")
#     # b0 = pm.Normal('b0_intercept', mu=mu_intercept, sigma=sigma_intercept, dims="session")
#     session_hierarchy_slope = pm.Normal('session_hierarchy_slope', mu=0.0, sigma=1.0, dims="session")
#     # b1 = pm.Normal('hierarchy_score', mu=mu_area, sigma=sigma_area, dims="session")
#     b_sign_rf = pm.Normal('b_sign_rf', mu=0, sigma=1)
#     # b1 = pm.Normal('sign_rf', mu=mu_rf, sigma=sigma_rf, dims="session")
#     b_log_fr = pm.Normal('b_log_fr', mu=0, sigma=1)
#     # b3 = pm.Normal('log_fr', mu=mu_log_fr, sigma=sigma_log_fr, dims="session")
#
#     # define linear model
#     yest = ( mu_intercept + session_intercept[session_idx] * sigma_intercept +
#              (mu_hierarchy_slope + session_hierarchy_slope[session_idx]*sigma_hierarchy_slope) * _data.hierarchy_score.values +
#              b_sign_rf * _data.sign_rf.values +
#              b_log_fr * _data.log_fr
#     )
#     # yest = ( b0[session_idx] +
#     #          b1[session_idx] * _data.sign_rf.values +
#     #          b2[session_idx] * _data.hierarchy_score.values +
#     #          b3[session_idx] * _data.log_fr
#     # )
#
#     # define normal likelihood with halfnormal error
#     # epsilon = pm.HalfCauchy('epsilon', 10)
#     epsilon = pm.HalfCauchy('epsilon', 1, dims = "session")
#     #likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon, observed=mx_en[ft_endog])
#     likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon[session_idx], observed=_data[ft_endog], dims="obs_id")
#     # likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon, observed=_data[ft_endog], dims="obs_id")
#     trc_session_specific_variance_model = pm.sample(2000, tune=2000, random_seed=seed)
#     print(pm.summary(trc_session_specific_variance_model).round(2))
#
# with pm.Model(coords=coords) as single_variance_model:
#     session_idx = pm.Data("session_idx", session_idxs, dims="obs_id")
#
#     # hyperpriors
#     mu_intercept = pm.Normal('mu_intercept', mu=0., sigma=1)
#     sigma_intercept = pm.HalfCauchy('sigma_intercept', beta=1)
#     # mu_rf = pm.Normal('mu_rf', mu=0., sigma=1)
#     # sigma_rf = pm.HalfCauchy('sigma_rf', beta=1)
#     # mu_log_fr = pm.Normal('mu_log_fr', mu=0., sigma=1)
#     # sigma_log_fr = pm.HalfCauchy('sigma_log_fr', beta=1)
#     mu_area = pm.Normal('mu_area', mu=0., sigma=1)
#     sigma_area = pm.HalfCauchy('sigma_area', beta=1)
#
#     b0 = pm.Normal('b0_intercept', mu=0.0, sigma=1.0, dims="session")
#     # b0 = pm.Normal('b0_intercept', mu=mu_intercept, sigma=sigma_intercept, dims="session")
#     b1 = pm.Normal('hierarchy_score', mu=0.0, sigma=1.0, dims="session")
#     # b1 = pm.Normal('hierarchy_score', mu=mu_area, sigma=sigma_area, dims="session")
#     b2 = pm.Normal('sign_rf', mu=0, sigma=1)
#     # b1 = pm.Normal('sign_rf', mu=mu_rf, sigma=sigma_rf, dims="session")
#     b3 = pm.Normal('log_fr', mu=0, sigma=1)
#     # b3 = pm.Normal('log_fr', mu=mu_log_fr, sigma=sigma_log_fr, dims="session")
#
#     # define linear model
#     yest = ( mu_intercept + b0[session_idx] * sigma_intercept +
#              (mu_area + b1[session_idx] * sigma_area) * _data.hierarchy_score.values +
#              b2 * _data.sign_rf.values +
#              b3 * _data.log_fr
#     )
#     # yest = ( b0[session_idx] +
#     #          b1[session_idx] * _data.sign_rf.values +
#     #          b2[session_idx] * _data.hierarchy_score.values +
#     #          b3[session_idx] * _data.log_fr
#     # )
#
#     # define normal likelihood with halfnormal error
#     epsilon = pm.HalfCauchy('epsilon', 10)
#     # epsilon = pm.HalfCauchy('epsilon', 1, dims = "session")
#     #likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon, observed=mx_en[ft_endog])
#     # likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon[session_idx], observed=_data[ft_endog], dims="obs_id")
#     likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon, observed=_data[ft_endog], dims="obs_id")
#     trc_single_variance_model = pm.sample(2000, tune=2000, random_seed=seed)
#
#     print(pm.summary(trc_single_variance_model).round(2))
#
# df_comp_loo = az.compare({"single_variance_model" : trc_single_variance_model, "session_specific_variance_model" : trc_session_specific_variance_model}, ic = "waic")
#
#
# az.plot_compare(df_comp_loo, insample_dev=False);
# pm.waic(trc_single_variance_model, single_variance_model)
# pm.waic(trc_session_specific_variance_model, session_specific_variance_model)

# %%
