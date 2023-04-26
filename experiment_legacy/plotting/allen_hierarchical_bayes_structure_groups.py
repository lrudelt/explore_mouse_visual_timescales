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

args = argparse.Namespace()
defined_measures = ["tau_C",
                    "tau_R",
                    "R_tot"]
defined_stimuli = ["movie",
                   "spontaneous"]

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


# args.measure = 'tau_C'
# __file__ = 'allen_hierarchical_bayes_structure_groups.py'

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
selected_structures = 'cortex+thalamus'
# selected_structures = 'cortex'

plot_stars = False # plot stars or p-values for tests

# setup analysis
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])
plot_settings['imgdir'] = '../img/'

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

# MODEL: STRUCTURAL GROUPS MODEL

model_name = "ho_categ_areas"

## set up model

ft_endog = measure
fts_num = ['log_fr'] # 'firing_rate'
fts_cat = ['sign_rf', 'structure_group', 'session_id']

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
                   ((data[fts_num] - data[fts_num].mean(0)) / (2 * data[fts_num].std(0)))),1)
_data = _data[_data[ft_endog] > -np.inf]

for var in _data.columns.values:
    _data = _data[~_data[var].isna()]

#fml_all = '{} ~ '.format(ft_endog) + ' + '.join(fts_num + fts_cat)

#(mx_en, mx_ex) = pt.dmatrices(fml_all, _data, return_type='dataframe')#, NA_action='raise')
# print(mx_ex.describe())
# print(mx_ex.columns.values)

_data.insert(5, 'higher_cortical', _data['structure_group'].apply(lambda x: x == 'higher cortical'))
_data.insert(6, 'thalamus', _data['structure_group'].apply(lambda x: x == 'Thalamus'))
_data.drop('structure_group', axis=1, inplace=True)

for var in _data.columns.values:
    _data[var] = _data[var].astype(theano.config.floatX)

session_idxs, sessions = pd.factorize(_data.session_id)
coords = {
    "session": sessions,
    "obs_id": np.arange(len(session_idxs))
}

with pm.Model(coords=coords) as model:
    session_idx = pm.Data("session_idx", session_idxs, dims="obs_id")

    # hyperpriors
    mu_intercept = pm.Normal('mu_intercept', mu=0., sigma=1)
    sigma_intercept = pm.HalfCauchy('sigma_intercept', beta=0.1)
    # mu_rf = pm.Normal('mu_rf', mu=0., sigma=1)
    # sigma_rf = pm.HalfCauchy('sigma_rf', beta=1)
    # mu_log_fr = pm.Normal('mu_log_fr', mu=0., sigma=1)
    # sigma_log_fr = pm.HalfCauchy('sigma_log_fr', beta=1)
    mu_higher_cortical_offset = pm.Normal('mu_higher_cortical_offset', mu=0., sigma=1)
    sigma_higher_cortical_offset = pm.HalfCauchy('sigma_higher_cortical_offset', beta=1)
    mu_thalamus_offset = pm.Normal('mu_thalamus_offset', mu=0., sigma=1)
    sigma_thalamus_offset = pm.HalfCauchy('sigma_thalamus_offset', beta=1)

    session_intercept = pm.Normal('session_intercept', mu=0.0, sigma=1.0, dims="session")
    # b0 = pm.Normal('b0_intercept', mu=mu_intercept, sigma=sigma_intercept, dims="session")
    session_higher_cortical_offset = pm.Normal('session_higher_cortical_offset', mu=0.0, sigma=1.0, dims="session")
    session_thalamus_offset = pm.Normal('session_thalamus_offset', mu=0.0, sigma=1.0, dims="session")
    # b1 = pm.Normal('hierarchy_score', mu=mu_area, sigma=sigma_area, dims="session")
    b_sign_rf = pm.Normal('b_sign_rf', mu=0, sigma=1)
    # b1 = pm.Normal('sign_rf', mu=mu_rf, sigma=sigma_rf, dims="session")
    b_log_fr = pm.Normal('b_log_fr', mu=0, sigma=1)
    # b3 = pm.Normal('log_fr', mu=mu_log_fr, sigma=sigma_log_fr, dims="session")

    # define linear model
    yest = ( mu_intercept + session_intercept[session_idx] * sigma_intercept +
             (mu_higher_cortical_offset + session_higher_cortical_offset[session_idx]*sigma_higher_cortical_offset) * _data['higher_cortical'].values +
             (mu_thalamus_offset + session_thalamus_offset[session_idx]*sigma_thalamus_offset) * _data['thalamus'].values +
             b_sign_rf * _data.sign_rf.values +
             b_log_fr * _data.log_fr.values
    )
    # define normal likelihood with halfnormal error
    epsilon = pm.HalfCauchy('epsilon', 10)
    if args.measure == "R_tot":
        likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon, observed=_data[ft_endog], dims="obs_id")
    else:
        alpha = pm.Normal('alpha', mu = 0., sigma=1)
        likelihood = pm.SkewNormal('likelihood', mu=yest, sigma=epsilon, alpha=alpha, observed=_data[ft_endog], dims="obs_id")


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
    eff_session_higher_cortical_offset = pm.Deterministic("eff_session_higher_cortical_offset", mu_higher_cortical_offset + session_higher_cortical_offset * sigma_higher_cortical_offset)
    eff_session_thalamus_offset = pm.Deterministic("eff_session_thalamus_offset",  mu_thalamus_offset + session_thalamus_offset * sigma_thalamus_offset)
    eff_session_intercept = pm.Deterministic("eff_session_intercept", mu_intercept + session_intercept * sigma_intercept)

## model visualization
pm.model_to_graphviz(model)

## prior predictive checks
#
with model:
    prior_pred_samples = pm.sample_prior_predictive(samples=1000, random_seed=seed)
#
# fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
# ax0.set_title('')
# utl.make_plot_pretty(ax0)
#
# _y = f_transf_int(prior_pred_samples['mu_intercept'] +
#                   prior_pred_samples['session_intercept'][:,0] * prior_pred_samples['sigma_intercept'] +
#                   prior_pred_samples['b_sign_rf'])
# # _y = f_transf_int(prior_pred_samples['mu_intercept'] +
# #                   prior_pred_samples['b_sign_rf'])
# np.median(_y)
#
# ax0.hist(_y[_y<1000], bins=30,
#         density=True, color = 'k', lw = 0, zorder = -1, rwidth=0.9)
# if measure == R_measure:
#     ax0.set_xlabel(r'$θ_0 \cdot θ_1^{rf}$')
# elif measure == T_measure or measure == C_measure:
#     ax0.set_xlabel(r'$θ_0 \cdot θ_1^{rf}$ (ms)')
#
# ax0.set_ylabel('ρ')
# if measure == T_measure:
#     ax0.set_xlim([-0.01,1010])
# elif measure == C_measure:
#     ax0.set_xlim([-0.01,10001])
# elif measure == R_measure:
#     ax0.set_xlim([-0.01, 1.01])
#
# utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_predictive", measure=args.measure)

## posterior sampling

with model:
    ## take samples
    trc = pm.sample(3000, tune=2000, random_seed=seed, target_accept = 0.95)

    print(pm.summary(trc).round(2))

# az.plot_trace(trc, compact = True, chain_prop ={"ls": "-"})

## posterior visualization (non-hierarchical parameters)

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
#'$θ_0$ (no rf, fir. rate={:.2f}Hz)'.format(np.expnp.mean(data['log_fr'])),
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

# # different way of getting the posterior data. For consistency use the one below
# with model:
#     fig0, ax0 = plt.subplots(1, 1, figsize=(plot_settings["panel_width"], 2))
#     fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=1)
#
#     utl.plot_posterior(trc[-500:], var_names=['mu_higher_cortical_offset'],
#                        ax=ax0, point_estimate='median',
#                        hdi_prob=0.95,
#                        transform=f_transf)
#
#     ax0.set_ylabel('p({$θ_3$} | E)')
#     ax0.set_xlabel('$θ_3$ (hier. score)')
#     ax0.set_title('')
#     utl.make_plot_pretty(ax0)
#
#     ax0.set_title(measure_name, ha='center')
#
# utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior_hierarchy_score", measure=args.measure)

idata = az.from_pymc3(
    trace=trc,
    prior=prior_pred_samples,
    model=model,
)

post = idata.posterior.assign_coords(session_idx=idata.constant_data.session_idx)

# offset higher_cortical

fig0 = plt.figure(figsize=(plot_settings["panel_width"]*1.7, 3))
# fig0.suptitle("offset higher cortical", ha = "center")
# fig0.suptitle(measure_name, ha = "center")
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)
fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=0.6, wspace=0.4)
ax1.axvline(x=0, ls = "--", lw =2, color = "0.0")    
ax3.axvline(x=1, ls = "--", lw =2, color = "0.0")    

with model:
    utl.plot_posterior(post["mu_higher_cortical_offset"].values[:,-1000:],
                       ax=ax1, point_estimate='median',
                       hdi_prob=0.95,
                       transform = f_transf_log)
    ax1.set_xlabel(r'mean offset $\mu_{\theta_{\mathrm{hc}}}$')
    ax1.set_ylabel(r'$p\left(\mu_{\theta_{\mathrm{hc}}} | E\right)$')
    utl.plot_posterior(post["sigma_higher_cortical_offset"].values[:,-1000:],
                       ax=ax2, point_estimate='median',
                       hdi_prob=0.95,
                       transform = f_transf_log)
    ax2.set_xlabel(r'std offset $\sigma_{\theta_{\mathrm{hc}}}$')
    ax2.set_ylabel(r'$p\left(\sigma_{\theta_{\mathrm{hc}}} | E\right)$')


    for session_idx in range(len(sessions)):
        utl.plot_posterior(post["eff_session_higher_cortical_offset"].values[:,-1000:,session_idx],
                       ax=ax3,
                       hdi = False,
                       color = sns.color_palette()[session_idx%10],
                       transform = f_transf)

    ax3.set_xlabel(r'higher cortical offset $\exp(\theta_{\mathrm{hc}})$')
    ax3.set_ylabel(r'$p\left(\theta_{\mathrm{hc}} | E\right)$')
    if args.measure == "tau_C":
        ax3.set_xlim([0.5,2.0])

for ax in [ax1,ax2,ax3]:
    utl.make_plot_pretty(ax)

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior_higher_cortical_offset", allen_bo=args.allen_bo, stimulus=args.stimulus, measure=args.measure)

# offset thalamus

fig0 = plt.figure(figsize=(plot_settings["panel_width"]*1.7, 3))
# fig0.suptitle(measure_name, ha = "center")
# fig0.suptitle("offset thalamus", ha = "center")
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)
fig0.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=0.6, wspace=0.4)
ax1.axvline(x=0, ls = "--", lw =2, color = "0.0")    
ax3.axvline(x=1, ls = "--", lw =2, color = "0.0")  

with model:
    utl.plot_posterior(post["mu_thalamus_offset"].values[:,-1000:],
                       ax=ax1, point_estimate='median',
                       hdi_prob=0.95,
                       transform = f_transf_log)
    ax1.set_xlabel(r'mean offset $\mu_{\theta_{\mathrm{th}}}$')
    ax1.set_ylabel(r'$p\left(\mu_{\theta_{\mathrm{th}}} | E\right)$')
    utl.plot_posterior(post["sigma_thalamus_offset"].values[:,-1000:],
                       ax=ax2, point_estimate='median',
                       hdi_prob=0.95,
                       transform = f_transf_log)
    ax2.set_xlabel(r'std offset $\sigma_{\theta_{\mathrm{th}}}$')
    ax2.set_ylabel(r'$p\left(\sigma_{\theta_{\mathrm{th}}} | E\right)$')


    for session_idx in range(len(sessions)):
        utl.plot_posterior(post["eff_session_thalamus_offset"].values[:,-1000:,session_idx],
                       ax=ax3,
                       hdi = False,
                       color = sns.color_palette()[session_idx%10],
                       transform = f_transf)

    ax3.set_xlabel(r'thalamus offset $\exp(\theta_{\mathrm{th}})$')
    ax3.set_ylabel(r'$p\left(\theta_{\mathrm{th}} | E\right)$')
    if args.measure == "tau_C":
        ax3.set_xlim([0.,2.0])

for ax in [ax1,ax2,ax3]:
    utl.make_plot_pretty(ax)

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior_thalamus_offset", allen_bo=args.allen_bo, stimulus=args.stimulus, measure=args.measure)

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

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior_intercept", allen_bo=args.allen_bo, stimulus=args.stimulus, measure=args.measure)
