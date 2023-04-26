import argparse

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

from sys import exit
from os.path import realpath, dirname, isfile, isdir
import os
import yaml

SCRIPT_DIR = dirname(realpath(__file__))
with open('{}/dirs.yaml'.format(SCRIPT_DIR), 'r') as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import patsy as pt
import scipy.stats as st
import arviz as az

import analysis_utils as utl

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors

seed = 12347

### settings

if args.allen_bo:
    analysis = 'allen_bo'
else:
    analysis = 'allen_fc'

center = 'median' # measure of central tendency
T_measure = 'log_tau_R'
R_measure = 'R_tot'
C_measure = 'mre_tau'

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

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_2500.csv".format(stats_dir, analysis)

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

# plot

# if args.allen_bo:
#     if args.measure == "tau_C":
#         measure = C_measure
#     elif args.measure == "tau_R":
#         measure = T_measure
#     elif args.measure == "R_tot":
#         measure = R_measure
# elif args.stimulus == "movie":
#     if args.measure == "tau_C":
#         measure = C_measure
#     elif args.measure == "tau_R":
#         measure = T_measure
#     elif args.measure == "R_tot":
#         measure = R_measure
# elif args.stimulus == "spontaneous":
#     if args.measure == "tau_C":
#         measure = C_measure
#     elif args.measure == "tau_R":
#         measure = T_measure
#     elif args.measure == "R_tot":
#         measure = R_measure
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
    measure_name = r'predictable information'# $R_{\mathregular{tot}}$'
    measure_name_short = r'$R_{\mathregular{tot}}$'

selection = utl.get_data_filter(data, measure)
data = data[selection]


## set up model

ft_endog = measure
fts_num = ['log_fr'] # 'firing_rate'
fts_cat = ['sign_rf', 'ecephys_structure_acronym']

# mean-centered & standardized according to Gelman: 1 / (2 * sd)
# Divide by 2 standard deviations in order to put the variance of a
# normally distributed variable nearer to the variance range of a
# binary variable. See
# http://www.stat.columbia.edu/~gelman/research/published/standardizing7.pdf
# for more info.

if measure == R_measure or measure == C_measure:
    ft_endog_data = np.log10(data[ft_endog])
elif measure == T_measure:
    ft_endog_data = data[ft_endog]

ft_endog_mean = ft_endog_data[ft_endog_data > -np.inf].mean(0)
ft_endog_sd = ft_endog_data[ft_endog_data > -np.inf].std(0)

_data = pd.concat(((ft_endog_data - ft_endog_mean) / ft_endog_sd,
                   data[fts_cat],
                   ((data[fts_num] - data[fts_num].mean(0)) / (2 * data[fts_num].std(0)))),1)
_data = _data[_data[ft_endog] > -np.inf]

fml_all = '{} ~ '.format(ft_endog) + ' + '.join(fts_num + fts_cat)

(mx_en, mx_ex) = pt.dmatrices(fml_all, _data, return_type='dataframe')#, NA_action='raise')
# print(mx_ex.describe())
# print(mx_ex.columns.values)

with pm.Model() as model:
    # define priors
    b0 = pm.Normal('b0_intercept', mu=0, sigma=1)
    b1 = pm.Normal('sign_rf[T.True]', mu=0, sigma=1)
    b2a = pm.Normal('ecephys_structure_acronym[T.LP]', mu=0, sigma=1)
    b2b = pm.Normal('ecephys_structure_acronym[T.VISal]', mu=0, sigma=1)
    b2c = pm.Normal('ecephys_structure_acronym[T.VISam]', mu=0, sigma=1)
    b2d = pm.Normal('ecephys_structure_acronym[T.VISl]', mu=0, sigma=1)
    b2e = pm.Normal('ecephys_structure_acronym[T.VISp]', mu=0, sigma=1)
    b2f = pm.Normal('ecephys_structure_acronym[T.VISpm]', mu=0, sigma=1)
    b2g = pm.Normal('ecephys_structure_acronym[T.VISrl]', mu=0, sigma=1)
    b3 = pm.Normal('log_fr', mu=0, sigma=1)

    # define linear model
    yest = ( b0 +
             b1 * mx_ex['sign_rf[T.True]'] +
             b2a * mx_ex['ecephys_structure_acronym[T.LP]'] +
             b2b * mx_ex['ecephys_structure_acronym[T.VISal]'] +
             b2c * mx_ex['ecephys_structure_acronym[T.VISam]'] +
             b2d * mx_ex['ecephys_structure_acronym[T.VISl]'] +
             b2e * mx_ex['ecephys_structure_acronym[T.VISp]'] +
             b2f * mx_ex['ecephys_structure_acronym[T.VISpm]'] +
             b2g * mx_ex['ecephys_structure_acronym[T.VISrl]'] +
             b3 * mx_ex['log_fr']# +
    )

    # define normal likelihood with halfnormal error
    epsilon = pm.HalfCauchy('epsilon', 10)
    likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon, observed=mx_en[ft_endog])

if measure == R_measure:
    f_transf_int = lambda x: 10**(x * ft_endog_sd + ft_endog_mean)
    f_transf = lambda x: 10**(x * ft_endog_sd)

    f_transf_int_inv = lambda x: (np.log10(x) - ft_endog_mean) / ft_endog_sd
elif measure == T_measure or measure == C_measure:
    f_transf_int = lambda x: 10**(x * ft_endog_sd + ft_endog_mean) * 1000
    f_transf = lambda x: 10**(x * ft_endog_sd)

    f_transf_int_inv = lambda x: (np.log10(np.divide(x, 1000)) - ft_endog_mean) / ft_endog_sd

with model:
    diff_LGN_V1 = pm.Deterministic("diff_LGN_V1", - b2e)
    diff_LGN_LM = pm.Deterministic("diff_LGN_LM", - b2d)
    diff_LGN_RL = pm.Deterministic("diff_LGN_RL", - b2g)
    diff_LGN_AL = pm.Deterministic("diff_LGN_AL", - b2b)
    diff_LGN_PM = pm.Deterministic("diff_LGN_PM", - b2f)
    diff_LGN_AM = pm.Deterministic("diff_LGN_AM", - b2c)
    diff_LGN_LP = pm.Deterministic("diff_LGN_LP", - b2a)

    diff_LM_V1 = pm.Deterministic("diff_LM_V1", b2d - b2e)
    diff_LM_RL = pm.Deterministic("diff_LM_RL", b2d - b2g)
    diff_LM_AL = pm.Deterministic("diff_LM_AL", b2d - b2b)
    diff_LM_PM = pm.Deterministic("diff_LM_PM", b2d - b2f)
    diff_LM_AM = pm.Deterministic("diff_LM_AM", b2d - b2c)
    diff_LM_LP = pm.Deterministic("diff_LM_LP", b2d - b2a)

    diff_RL_V1 = pm.Deterministic("diff_RL_V1", b2g - b2e)
    diff_RL_AL = pm.Deterministic("diff_RL_AL", b2g - b2b)
    diff_RL_PM = pm.Deterministic("diff_RL_PM", b2g - b2f)
    diff_RL_AM = pm.Deterministic("diff_RL_AM", b2g - b2c)
    diff_RL_LP = pm.Deterministic("diff_RL_LP", b2g - b2a)

    diff_AL_V1 = pm.Deterministic("diff_AL_V1", b2b - b2e)
    diff_AL_PM = pm.Deterministic("diff_AL_PM", b2b - b2f)
    diff_AL_AM = pm.Deterministic("diff_AL_AM", b2b - b2c)
    diff_AL_LP = pm.Deterministic("diff_AL_LP", b2b - b2a)

    diff_PM_V1 = pm.Deterministic("diff_PM_V1", b2f - b2e)
    diff_PM_AM = pm.Deterministic("diff_PM_AM", b2f - b2c)
    diff_PM_LP = pm.Deterministic("diff_PM_LP", b2f - b2a)

    diff_AM_V1 = pm.Deterministic("diff_AM_V1", b2c - b2e)
    diff_AM_LP = pm.Deterministic("diff_AM_LP", b2c - b2a)

    diff_LP_V1 = pm.Deterministic("diff_LP_V1", b2a - b2e)


## model visualization

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
az.plot_dist(f_transf_int(b0.random(size=5000)))
if measure == R_measure:
    ax0.set_xlabel('$θ_0$')
elif measure == T_measure or measure == C_measure:
    ax0.set_xlabel('$θ_0$ (ms)')
ax0.set_ylabel('$p(θ_0)$')
ax0.set_title('')
utl.make_plot_pretty(ax0)

utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_intercept", stimulus=args.stimulus, measure=args.measure)

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
az.plot_dist(b1.random(size=5000))
ax0.set_xlabel('log($θ_1$)')
ax0.set_ylabel('$p($log($θ_1))$')
ax0.set_title('')
utl.make_plot_pretty(ax0)

ax0.text(1.1, 0.28,
         r'$\mathcal{N}(\mu=0,\sigma=1)$', fontsize=8)

utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_coeff", stimulus=args.stimulus, measure=args.measure)


## prior predictive checks

with model:
    prior_pred_samples = pm.sample_prior_predictive(samples=500, random_seed=seed)

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
ax0.set_title('')
utl.make_plot_pretty(ax0)

_y = f_transf_int(prior_pred_samples['b0_intercept']
                  + prior_pred_samples['ecephys_structure_acronym[T.VISp]']
                  + prior_pred_samples['sign_rf[T.True]'])

ax0.hist(_y, bins=30,
        density=True, color = 'k', lw = 0, zorder = -1, rwidth=0.9)
if measure == R_measure:
    ax0.set_xlabel(r'$θ_0 \cdot θ_1^{rf} \cdot θ_2^{V1}$')
elif measure == T_measure or measure == C_measure:
    ax0.set_xlabel(r'$θ_0 \cdot θ_1^{rf} \cdot θ_2^{V1}$ (ms)')

ax0.set_ylabel('ρ')
if measure == T_measure:
    ax0.set_xlim([-0.01,1010])
elif measure == C_measure:
    ax0.set_xlim([-0.01,10001])
elif measure == R_measure:
    ax0.set_xlim([-0.01, 1.01])

utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_predictive", stimulus=args.stimulus, measure=args.measure)


## posterior sampling

with model:
    ## take samples
    trc = pm.sample(1000, random_seed=seed)

    print(pm.summary(trc).round(2))


## posterior visualization

with model:
    fig1, axes = plt.subplots(4, 1, figsize=(plot_settings["panel_width"], 6))
    fig1.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=1)

    utl.plot_posterior(trc[-250:], var_names=['b0_intercept'],
                       ax=axes[0], point_estimate='median',
                       hdi_prob=0.95,
                       transform=f_transf_int)

    utl.plot_posterior(trc[-250:], var_names=[#'b0_intercept',
        'sign_rf[T.True]',
        #'stimulus[T.spontaneous]',
        #'sign_rf[T.True]:stimulus[T.spontaneous]',
        # 'firing_rate',
                                             'log_fr',
        'epsilon'],
                       ax=axes[1:], point_estimate='median',
                       hdi_prob=0.95,
                       transform=f_transf)

    for ax, var, var_name in zip(axes,
                                 ['$θ_0$',
                                  '$θ_1$',
                                  '$θ_2$',
                                  'ε'],
                                 ['$θ_0$ (no rf, fir. rate={:.2f}Hz)'.format(10**np.mean(data['log_fr'])),
                                  '$θ_1$ (rf)',
                                  '$θ_2$ (log fir. rate)',
                                  'ε']):

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

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior", stimulus=args.stimulus, measure=args.measure)


with model:
    fig1, axes = plt.subplots(6, 4, figsize=(plot_settings["textwidth"], 10))
    fig1.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=1, wspace=0.2)

    utl.plot_posterior(trc[-250:], var_names=["diff_LGN_V1",
                                              "diff_LGN_LM",
                                              "diff_LGN_RL",
                                              "diff_LGN_AL",
                                              "diff_LGN_PM",
                                              "diff_LGN_AM",
                                              "diff_LGN_LP",
                                              "diff_LM_V1",
                                              "diff_LM_RL",
                                              "diff_LM_AL",
                                              "diff_LM_PM",
                                              "diff_LM_AM",
                                              "diff_LM_LP",
                                              "diff_RL_V1",
                                              "diff_RL_AL",
                                              "diff_RL_PM",
                                              "diff_RL_AM",
                                              "diff_RL_LP",
                                              "diff_AL_V1",
                                              "diff_AL_PM",
                                              "diff_AL_AM",
                                              "diff_AL_LP",
                                              "diff_PM_V1",
                                              "diff_PM_AM",
                                              "diff_PM_LP",
                                              "diff_AM_V1",
                                              "diff_AM_LP",
                                              "diff_LP_V1"],
                       ax=axes, point_estimate='median',
                       hdi_prob=0.95,
                       transform=f_transf)

    for ax, title in zip(axes.flatten(),
                         ["LGN-V1",
                          "LGN-LM",
                          "LGN-RL",
                          "LGN-AL",
                          "LGN-PM",
                          "LGN-AM",
                          "LGN-LP",
                          "LM-V1",
                          "LM-RL",
                          "LM-AL",
                          "LM-PM",
                          "LM-AM",
                          "LM-LP",
                          "RL-V1",
                          "RL-AL",
                          "RL-PM",
                          "RL-AM",
                          "RL-LP",
                          "AL-V1",
                          "AL-PM",
                          "AL-AM",
                          "AL-LP",
                          "PM-V1",
                          "PM-AM",
                          "PM-LP",
                          "AM-V1",
                          "AM-LP",
                          "LP-V1"]):
        ax.set_title(title)
        utl.make_plot_pretty(ax)

    fig1.suptitle(measure_name)

    for ax in axes[5]:
        ax.set_xlabel(r'$\theta_{3,i}/\theta_{3,j}$')

    for axes_row in axes:
        axes_row[0].set_ylabel(r'p($\theta_{3,i}/\theta_{3,j}$ | E)')

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior_areas", stimulus=args.stimulus, measure=args.measure)


## posterior predictive checks

with model:
    ppc = pm.sample_posterior_predictive(
        trc,
        random_seed=seed,
    )

idata = az.from_pymc3(
    trace=trc,
    prior=prior_pred_samples,
    posterior_predictive=ppc,
    model=model,
)

fig0 = plt.figure(figsize=(0.6*plot_settings["textwidth"],3))
ax_ppc = fig0.add_subplot(211)
ax1 = fig0.add_subplot(212)
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1, hspace=0.8)

# cf https://oriolabrilpla.cat/python/arviz/pymc3/2019/07/31/loo-pit-tutorial.html
def plot_ppc_loopit_2p(idata, title, ax_ppc=None, ax1=None):
    if np.any(np.array([ax_ppc, ax1]) is None):
        fig = plt.figure(figsize=(12,9))
        ax_ppc = fig.add_subplot(121)
        ax1 = fig.add_subplot(122);
    az.plot_ppc(idata, ax=ax_ppc, num_pp_samples=50, random_seed=seed);
    az.plot_loo_pit(idata, y='likelihood', ecdf=False, ax=ax1);
    ax_ppc.set_title(title)
    ax_ppc.set_xlabel("")
    return np.array([ax_ppc, ax1])

plot_ppc_loopit_2p(idata, measure_name, ax_ppc, ax1);

utl.make_plot_pretty(ax_ppc)
utl.make_plot_pretty(ax1)

def format_label_transf(x, pos):
    x10 = f_transf_int(x)
    if x10 >= 1:
        return "{:.0f}".format(x10)
    if x10 >= 0.1:
        return "{:.1f}".format(x10)
    elif x10 >= 0.01:
        return "{:.2f}".format(x10)
    elif x10 >= 0.001:
        return "{:.3f}".format(x10)

ax_ppc.legend(loc=(0.74, 0.4))
ax1.legend(loc=(1.1, 0.01))

for legend in [ax_ppc.get_legend(),
               ax1.get_legend()]:
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')

ax_ppc.xaxis.set_major_formatter(ticker.FuncFormatter(format_label_transf))
ax_ppc.set_xlabel(measure_name)
ax_ppc.set_ylabel('p({})'.format(measure_name_short))
ax_ppc.set_xlim([-5, 5])

#R_transf_inv = lambda x: (np.log10(x) - R_mean) / R_sd
#for k in [0.01, 0.1, 1]:
#    print(R_transf_inv(k))
if measure == T_measure:
    # ax_ppc.set_xticks([-4.374360930313099, -1.7244322830891257, 0.9254963641348477, 3.575425011358821])
    ax_ppc.set_xticks(f_transf_int_inv([1, 10, 100, 1000]))
elif measure == C_measure:
    # ax_ppc.set_xticks([-3.6484831459912854, -1.3205235746152693, 1.0074359967607471, 3.335395568136763])
    ax_ppc.set_xticks(f_transf_int_inv([10, 100, 1000, 10000]))
elif measure == R_measure:
    #ax_ppc.set_xticks([-2.8483186093823685, 0.6190576894032882, 4.086433988188944])
    ax_ppc.set_xticks(f_transf_int_inv([0.01, 0.1, 1.0]))

ax1.set_xlabel(r'$\mathregular{Pr}(y_i^{\mathregular{rep}} \leq y_i|y_{-i})$')
ax1.set_ylabel(r'$p(\mathregular{Pr}(y_i^{\mathregular{rep}} \leq y_i|y_{-i}))$')

utl.save_plot(plot_settings, f"{__file__[:-3]}_ppc_loopit", stimulus=args.stimulus, measure=args.measure)

az.loo(idata) # normal likelihood on log scale


## correlation with hierarchy score

measure_per_structure = {}
measure_intercept = trc['b0_intercept']
for structure in structures:
    if structure in structures_cortex:
        measure_per_structure[structure] \
        = f_transf_int(measure_intercept
                       + trc['ecephys_structure_acronym[T.{}]'.format(structure)])
    # if structure == 'LGd':
    #     measure_per_structure[structure] = f_transf_int(measure_intercept)
    # else:
    #     measure_per_structure[structure] \
    #     = f_transf_int(measure_intercept
    #                    + trc['ecephys_structure_acronym[T.{}]'.format(structure)])


fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])

x = []
y = []

for structure in structures:
    # _x = structures_map[structure]['hierarchy_score']
    # # _y = measure_per_structure[structure]
    # _y = np.median(measure_per_structure[structure])

    if structure in structures_cortex:
        x += [structures_map[structure]['hierarchy_score']]
        y += [np.median(measure_per_structure[structure])]

utl.plot_hdi(measure_per_structure, structures_map=structures_map, ax=ax0)

ax0.set_ylabel(measure_name_short)
ax0.set_xlabel('hierarchy score')

slope,intercept,r,p,std = st.linregress(x, y)
x2 = np.linspace(min(x), max(x), 10)

ax0.plot(x2, x2*slope+intercept, '--k', alpha=0.5)

r_s, p_s = st.spearmanr(x, y)
r_p, p_p = st.pearsonr(x, y)

text = '$r_P$ = {:.2f}; $P_P$ = {:.3f}\n$r_S$ = {:.2f}; $P_S$ = {:.3f}'.format(r_p,
                                                                               p_p,
                                                                               r_s,
                                                                               p_s)
if measure == T_measure:
    lims = [29, 64]
    xpos = -0.2
    ypos = 31
elif measure == C_measure:
    lims = [190, 620]
    xpos = -0.2
    ypos = 220
elif measure == R_measure:
    lims = [0.066, 0.093]
    xpos = -0.2
    ypos = 0.067

ax0.text(xpos, ypos,
         text, fontsize=8)
ax0.set_title(measure_name)
ax0.set_ylim(lims)
utl.make_plot_pretty(ax0)
ax0.set_xticks([-0.5,-0.25,0,0.25])

# legend
# legend_elements = [Line2D([0], [0], marker='s', color=structures_map[structure]['color'],
#                           label=structures_map[structure]['name'],
#                           markerfacecolor=structures_map[structure]['color'],
#                           ms=10,
#                           linewidth=0)
#                    for structure in structures]
# legend = ax0.legend(handles=legend_elements,
#                     fancybox=False,
#                     loc="lower right",
#                     bbox_to_anchor=(1.40, 0.1))
# frame = legend.get_frame()
# frame.set_facecolor('0.9')
# frame.set_edgecolor('0.9')


utl.save_plot(plot_settings, f"{__file__[:-3]}_hierarchy", stimulus=args.stimulus, measure=args.measure)

measure_per_structure = {}
measure_intercept = trc['b0_intercept']
for structure in structures:
    if structure == 'LGd':
        measure_per_structure[structure] = f_transf_int(measure_intercept)
    else:
        measure_per_structure[structure] \
        = f_transf_int(measure_intercept
                       + trc['ecephys_structure_acronym[T.{}]'.format(structure)])


## areas grouped

fts_cat = ['sign_rf', 'structure_group']

_data = pd.concat(((ft_endog_data - ft_endog_mean) / ft_endog_sd,
                   data[fts_cat],
                   ((data[fts_num] - data[fts_num].mean(0)) / (2 * data[fts_num].std(0)))),1)
_data = _data[_data[ft_endog] > -np.inf]

fml_all = '{} ~ '.format(ft_endog) + ' + '.join(fts_num + fts_cat)

(mx_en, mx_ex) = pt.dmatrices(fml_all, _data, return_type='dataframe')#, NA_action='raise')

with pm.Model() as sg_model:
    # define priors
    b0 = pm.Normal('b0_intercept', mu=0, sigma=1)
    b1 = pm.Normal('sign_rf[T.True]', mu=0, sigma=1)
    b2a = pm.Normal('structure_group[T.V1]', mu=0, sigma=1)
    b2b = pm.Normal('structure_group[T.higher cortical]', mu=0, sigma=1)
    b3 = pm.Normal('log_fr', mu=0, sigma=1)

    # define linear model
    yest = ( b0 +
             b1 * mx_ex['sign_rf[T.True]'] +
             b2a * mx_ex['structure_group[T.V1]'] +
             b2b * mx_ex['structure_group[T.higher cortical]'] +
             b3 * mx_ex['log_fr']# +
    )

    # define normal likelihood with halfnormal error
    epsilon = pm.HalfCauchy('epsilon', 10)
    likelihood = pm.Normal('likelihood', mu=yest, sd=epsilon, observed=mx_en[ft_endog])


with sg_model:
    diff_thalamus_V1 = pm.Deterministic("diff_thalamus_V1", - b2a)
    diff_thalamus_higher = pm.Deterministic("diff_thalamus_higher", - b2b)

    diff_V1_higher = pm.Deterministic("diff_V1_higher", b2a - b2b)

## model visualization

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
az.plot_dist(f_transf_int(b0.random(size=5000)))
if measure == R_measure:
    ax0.set_xlabel('$θ_0$')
elif measure == T_measure or measure == C_measure:
    ax0.set_xlabel('$θ_0$ (ms)')
ax0.set_ylabel('$p(θ_0)$')
ax0.set_title('')
utl.make_plot_pretty(ax0)

utl.save_plot(plot_settings, f"{__file__[:-3]}_sg_prior_intercept", stimulus=args.stimulus, measure=args.measure)

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
az.plot_dist(b1.random(size=5000))
ax0.set_xlabel('log($θ_1$)')
ax0.set_ylabel('$p($log($θ_1))$')
ax0.set_title('')
utl.make_plot_pretty(ax0)

ax0.text(1.1, 0.28,
         r'$\mathcal{N}(\mu=0,\sigma=1)$', fontsize=8)

utl.save_plot(plot_settings, f"{__file__[:-3]}_sg_prior_coeff", stimulus=args.stimulus, measure=args.measure)


## prior predictive checks

with sg_model:
    prior_pred_samples = pm.sample_prior_predictive(samples=500, random_seed=seed)

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
ax0.set_title('')
utl.make_plot_pretty(ax0)

_y = f_transf_int(prior_pred_samples['b0_intercept']
                  + prior_pred_samples['structure_group[T.V1]']
                  + prior_pred_samples['sign_rf[T.True]'])

ax0.hist(_y, bins=30,
        density=True, color = 'k', lw = 0, zorder = -1, rwidth=0.9)
if measure == R_measure:
    ax0.set_xlabel(r'$θ_0 \cdot θ_1^{rf} \cdot θ_2^{V1}$')
elif measure == T_measure or measure == C_measure:
    ax0.set_xlabel(r'$θ_0 \cdot θ_1^{rf} \cdot θ_2^{V1}$ (ms)')

ax0.set_ylabel('ρ')
if measure == T_measure:
    ax0.set_xlim([-0.01,1010])
elif measure == C_measure:
    ax0.set_xlim([-0.01,10001])
elif measure == R_measure:
    ax0.set_xlim([-0.01, 1.01])

utl.save_plot(plot_settings, f"{__file__[:-3]}_sg_prior_predictive", stimulus=args.stimulus, measure=args.measure)


## posterior sampling

with sg_model:
    ## take samples
    trc = pm.sample(1000, random_seed=seed)

    print(pm.summary(trc).round(2))


## posterior visualization

with sg_model:
    fig1, axes = plt.subplots(4, 1, figsize=(plot_settings["panel_width"], 6))
    fig1.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=1)

    utl.plot_posterior(trc[-250:], var_names=['b0_intercept'],
                       ax=axes[0], point_estimate='median',
                       hdi_prob=0.95,
                       transform=f_transf_int)

    utl.plot_posterior(trc[-250:], var_names=[#'b0_intercept',
        'sign_rf[T.True]',
        #'stimulus[T.spontaneous]',
        #'sign_rf[T.True]:stimulus[T.spontaneous]',
        # 'firing_rate',
                                             'log_fr',
        'epsilon'],
                       ax=axes[1:], point_estimate='median',
                       hdi_prob=0.95,
                       transform=f_transf)

    for ax, var, var_name in zip(axes,
                                 ['$θ_0$',
                                  '$θ_1$',
                                  '$θ_2$',
                                  'ε'],
                                 ['$θ_0$ (no rf, fir. rate={:.2f}Hz)'.format(10**np.mean(data['log_fr'])),
                                  '$θ_1$ (rf)',
                                  '$θ_2$ (log fir. rate)',
                                  'ε']):

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

utl.save_plot(plot_settings, f"{__file__[:-3]}_sg_posterior", stimulus=args.stimulus, measure=args.measure)


with sg_model:
    fig1, axes = plt.subplots(3, 1, figsize=(plot_settings["textwidth"], 5))
    fig1.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1, hspace=1, wspace=0.2)

    utl.plot_posterior(trc[-250:], var_names=["diff_thalamus_V1",
                                              "diff_thalamus_higher",
                                              "diff_V1_higher"],
                       ax=axes, point_estimate='median',
                       hdi_prob=0.95,
                       transform=f_transf)

    for ax, title in zip(axes.flatten(),
                         ["Thalamus-V1",
                          "Thalamus-Higher Cortical",
                          "V1-Higher Cortical"]):
        ax.set_title(title)
        utl.make_plot_pretty(ax)

    fig1.suptitle(measure_name)

    axes[2].set_xlabel(r'$\theta_{3,i}/\theta_{3,j}$')

    for ax in axes:
        ax.set_ylabel(r'p($\theta_{3,i}/\theta_{3,j}$ | E)')

utl.save_plot(plot_settings, f"{__file__[:-3]}_sg_posterior_areas", stimulus=args.stimulus, measure=args.measure)


## posterior predictive checks

with sg_model:
    ppc = pm.sample_posterior_predictive(
        trc,
        random_seed=seed,
    )

idata = az.from_pymc3(
    trace=trc,
    prior=prior_pred_samples,
    posterior_predictive=ppc,
    model=sg_model,
)

fig0 = plt.figure(figsize=(0.6*plot_settings["textwidth"],3))
ax_ppc = fig0.add_subplot(211)
ax1 = fig0.add_subplot(212)
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1, hspace=0.8)

plot_ppc_loopit_2p(idata, measure_name, ax_ppc, ax1);

utl.make_plot_pretty(ax_ppc)
utl.make_plot_pretty(ax1)

# ax_ppc.legend(loc=(0.74, 0.4))
# ax1.legend(loc=(1.1, 0.01))
#
# for legend in [ax_ppc.get_legend(),
#                ax1.get_legend()]:
#     frame = legend.get_frame()
#     frame.set_facecolor('0.9')
#     frame.set_edgecolor('0.9')

ax_ppc.xaxis.set_major_formatter(ticker.FuncFormatter(format_label_transf))
ax_ppc.set_xlabel(measure_name_short)
ax_ppc.set_ylabel('p({})'.format(measure_name_short))
ax_ppc.set_xlim([-5, 5])

#R_transf_inv = lambda x: (np.log10(x) - R_mean) / R_sd
#for k in [0.01, 0.1, 1]:
#    print(R_transf_inv(k))
if measure == T_measure:
    # ax_ppc.set_xticks([-4.374360930313099, -1.7244322830891257, 0.9254963641348477, 3.575425011358821])
    ax_ppc.set_xticks(f_transf_int_inv([1, 10, 100, 1000]))
elif measure == C_measure:
    # ax_ppc.set_xticks([-3.6484831459912854, -1.3205235746152693, 1.0074359967607471, 3.335395568136763])
    ax_ppc.set_xticks(f_transf_int_inv([10, 100, 1000, 10000]))
elif measure == R_measure:
    # ax_ppc.set_xticks([-2.8483186093823685, 0.6190576894032882, 4.086433988188944])
    ax_ppc.set_xticks(f_transf_int_inv([0.01, 0.1, 1.0]))


ax1.set_xlabel(r'$\mathregular{Pr}(y_i^{\mathregular{rep}} \leq y_i|y_{-i})$')
ax1.set_ylabel(r'$p(\mathregular{Pr}(y_i^{\mathregular{rep}} \leq y_i|y_{-i}))$')

utl.save_plot(plot_settings, f"{__file__[:-3]}_sg_ppc_loopit", stimulus=args.stimulus, measure=args.measure)

az.loo(idata) # normal likelihood on log scale


structure_group_names = ["Thalamus", "V1", "higher cortical"]
structure_group_components = [['LGd','LP'],
                              ['VISp'],
                              ['VISl','VISrl','VISal','VISpm','VISam']]


measure_per_structure_group = {}
measure_intercept = trc['b0_intercept']
for structure_group_name, _structure_group_components in zip(structure_group_names,
                                                             structure_group_components):
    if structure_group_name == 'Thalamus':
        measure_per_structure_group[structure_group_name] = f_transf_int(measure_intercept)
    else:
        measure_per_structure_group[structure_group_name] \
        = f_transf_int(measure_intercept
                       + trc['structure_group[T.{}]'.format(structure_group_name)])

fig0, ax = plt.subplots(figsize=plot_settings["panel_size"])
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)

if args.allen_bo:
    if args.measure == "tau_C":
        lims = [149, 960]
        ax.set_title('intrinsic timescale')
    elif args.measure == "tau_R":
        lims = [17, 106]
        ax.set_title('information timescale')
    elif args.measure == "R_tot":
        lims = [0.047, 0.115]
        ax.set_title('predictable information')
elif args.stimulus == "movie":
    if args.measure == "tau_C":
        lims = [149, 960]
        ax.set_title('intrinsic timescale')
    elif args.measure == "tau_R":
        lims = [17, 86]
        ax.set_title('information timescale')
    elif args.measure == "R_tot":
        lims = [0.057, 0.125]
        ax.set_title('predictable information')
elif args.stimulus == "spontaneous":
    if args.measure == "tau_C":
        lims = [149, 960]
        ax.set_title('intrinsic timescale')
    elif args.measure == "tau_R":
        lims = [19, 89]
        ax.set_title('information timescale')
    elif args.measure == "R_tot":
        lims = [0.043, 0.096]
        ax.set_title('predictable information')


x_pos = [0, 0.8, 2]
y_pos = max([max(v) for v in measure_per_structure.values()])
diff_names = ['thalamus', 'V1', 'higher']

_xticks = []
_xlabels = []

for _x, structure_group_name, _structure_group_components, color in zip(x_pos,
                                                                        structure_group_names,
                                                                        structure_group_components,
                                                                        ['0.2', '0.2', '0.2']):

    values = az.hdi(measure_per_structure_group[structure_group_name], 0.95, multimodal=False)
    _y = np.median(measure_per_structure_group[structure_group_name])

    ax.plot([_x]*len(values),
            values,
            lw=4,
            color=color,
            label=structure_group_name,
    )
    ax.plot(_x,
            _y,
            's',
            ms=8,
            mew=2,
            color="white",
            mec=color
    )


    if len(_structure_group_components) > 1:
        hs = sorted([structures_map[structure]['hierarchy_score'] for structure in _structure_group_components])
        hs_rank = {}
        for i, _hs in enumerate(hs):
            hs_rank[_hs] = i

        for structure in _structure_group_components:
            values = az.hdi(measure_per_structure[structure], 0.95, multimodal=False)
            _y = np.median(measure_per_structure[structure])

            if structure_group_name == 'Thalamus':
                _width = 0.6
            else:
                _width = 0.8
            __x = _x - _width/2 + _width*hs_rank[structures_map[structure]['hierarchy_score']] / (len(hs) - 1)
            if __x >= _x and len(hs) % 2 == 1:
                __x += _width / (len(hs) - 1)

            ax.plot([__x]*len(values),
                    values,
                    lw=2,
                    alpha=.6,
                    color=structures_map[structure]['color'],
            )
            ax.plot(__x,
                    _y,
                    'o',
                    color="white",
                    mec=matplotlib.colors.colorConverter.to_rgba(structures_map[structure]['color'], alpha=.6)
            )

            _xticks += [__x]
            _xlabels += [structures_map[structure]['name']]
    else:
        structure = _structure_group_components[0]

        values = az.hdi(measure_per_structure[structure], 0.95, multimodal=False)
        _y = np.median(measure_per_structure[structure])

        __x = _x + 0.2

        ax.plot([__x]*len(values),
                values,
                lw=2,
                alpha=.6,
                color=structures_map[structure]['color'],
        )
        ax.plot(__x,
                _y,
                'o',
                color="white",
                mec=matplotlib.colors.colorConverter.to_rgba(structures_map[structure]['color'], alpha=.6)
        )


        _xticks += [_x]
        _xlabels += [structures_map[structures[0]]['name']]


for i, j, y_off_f, x_off_i, x_off_j in zip([0, 1, 0],
                                           [1, 2, 2],
                                           [0, 1, 2],
                                           [0, 0.05, 0],
                                           [-0.05, 0, 0]):
    diff_name = "diff_{}_{}".format(diff_names[i], diff_names[j])
    signs = np.sign(az.hdi(trc[diff_name], 0.95, multimodal=False))

    if signs[0] == signs[1]:
        x1, x2 = x_pos[i] + x_off_i, x_pos[j] + x_off_j
        w = 0.05 * (lims[1] - lims[0])

        y = y_pos + 1.5 * w

        y_off = y_off_f * w * 2
        y += y_off

        text = "*"
        text_y_pos = y+2*w

        # ax.plot([x1, x1, x2, x2], [y, y+w, y+w, y], lw=1.5, c='k')
        ax.plot([x1, x2], [y+w, y+w], lw=1.5, c='k')
        ax.text((x1+x2)/2, text_y_pos, text, ha='center', va='top', color='k')

        # #ax.plot([x1, x1, x2, x2], [y, y+w, y+w, y], lw=1.5, c='k')
        # ax.plot([x1, x2], [y+w, y+w], lw=1.5, c='k')
        # ax.text((x1+x2)/2, y+2*w, "*"*num_stars, ha='center', va='top', color='k')

#ax.set_xticks(_xticks)
#ax.set_xticklabels(_xlabels)
ax.set_xticks(x_pos)
ax.set_xticklabels(structure_group_names)
ax.set_ylim(lims)

ax.set_ylabel(measure_name)
utl.make_plot_pretty(ax)

utl.save_plot(plot_settings, f"{__file__[:-3]}_sg_grouped", stimulus=args.stimulus, measure=args.measure)
