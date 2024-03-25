import argparse

defined_measures = ["tau_C",
                    "tau_R",
                    "R_tot"]
parser = argparse.ArgumentParser()
parser.add_argument('measure', type=str, help=f'one of {defined_measures}')
args = parser.parse_args()
if not args.measure in defined_measures:
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

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors
from matplotlib import cm

seed = 12347

### settings
analysis = 'brian'

center = 'median' # measure of central tendency
T_measure = 'log_tau_R'
R_measure = 'R_tot'
C_measure = 'mre_tau'

if args.measure == "tau_C":
    T0 = 0.03 # 30 ms
else:
    T0 = 0.005 # 5 ms

kins = [ 68.,  72.,  76.,  80.,  84.,  88.,  92.,  96., 100., 104., 108.]
    
plot_stars = False # plot stars or p-values for tests

# setup analysis
plot_settings = utl.get_default_plot_settings()
plt.rcParams.update(plot_settings['rcparams'])

stats_dir = dir_settings['stats_dir']

csv_file_name = "{}/{}_statistics_T0_05.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_750.h5".format(stats_dir, analysis)
m_ar_stats_file_name = "{}/{}_m_ar_statistics.csv".format(stats_dir, analysis)

### import data

data = utl.get_analysis_data(csv_file_name, analysis,
                             mre_stats_file_name=mre_stats_file_name,
                             m_ar_stats_file_name=m_ar_stats_file_name)
data = data[data['kin'].isin(kins)]

if not args.measure == "tau_C":
    # we analysed 25 neurons per network for predictable information
    # but all 512 for the intrinsic timescale analysis
    data = data[~np.isnan(data['R_tot'])]
    data = data[utl.df_filter(data, T0=T0)]

# make sure data as expected
try:
    num_neurons = utl.get_expected_number_of_neurons(analysis,
                                                     measure=args.measure)
    assert np.isclose(len(data), num_neurons, atol=100)
except:
    print(f'number of neurons is {len(data)}, expected {num_neurons}')
    exit()

# plot

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

selection = utl.get_data_filter(data, measure, analysis=analysis)
data = data[selection]

## set up model

ft_endog = measure
fts_num = ['log_fr', 'm_ar']
fts_cat = []

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
    b1 = pm.Normal('log_fr', mu=0, sigma=1)
    b2 = pm.Normal('m_ar', mu=0, sigma=1)

    # define linear model
    yest = ( b0 +
             b1 * mx_ex['log_fr'] +
             b2 * mx_ex['m_ar']
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

utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_intercept", measure=args.measure)

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
az.plot_dist(b1.random(size=5000))
ax0.set_xlabel('log($θ_1$)')
ax0.set_ylabel('$p($log($θ_1))$')
ax0.set_title('')
utl.make_plot_pretty(ax0)

ax0.text(1.1, 0.28,
         r'$\mathcal{N}(\mu=0,\sigma=1)$', fontsize=8)

utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_coeff", measure=args.measure)


## prior predictive checks

with model:
    prior_pred_samples = pm.sample_prior_predictive(samples=500, random_seed=seed)

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])
ax0.set_title('')
utl.make_plot_pretty(ax0)

_y = f_transf_int(prior_pred_samples['b0_intercept']
                  + prior_pred_samples['log_fr'] * 0.2
                  + prior_pred_samples['m_ar'] * 0.4)

ax0.hist(_y, bins=30,
        density=True, color = 'k', lw = 0, zorder = -1, rwidth=0.9)
if measure == R_measure:
    ax0.set_xlabel(r'$θ_0 \cdot θ_1^{0.2} \cdot θ_2^{0.4}$')
elif measure == T_measure or measure == C_measure:
    ax0.set_xlabel(r'$θ_0 \cdot θ_1^{0.2} \cdot θ_2^{0.4}$ (ms)')

ax0.set_ylabel('ρ')
if measure == T_measure:
    ax0.set_xlim([-0.01,1010])
elif measure == C_measure:
    ax0.set_xlim([-0.01,100001])
elif measure == R_measure:
    ax0.set_xlim([-0.01, 1.01])

utl.save_plot(plot_settings, f"{__file__[:-3]}_prior_predictive", measure=args.measure)

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

    utl.plot_posterior(trc[-250:], var_names=['log_fr',
                                              'm_ar',
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
                                  '$θ_1$ (log fir. rate)',
                                  '$θ_2$ ($m_{ar}$)',
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

utl.save_plot(plot_settings, f"{__file__[:-3]}_posterior", measure=args.measure)


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
    ax_ppc.set_xticks(f_transf_int_inv([1, 10]))
elif measure == C_measure:
    #ax_ppc.set_xticks([-3.6484831459912854, -1.3205235746152693, 1.0074359967607471, 3.335395568136763])
    ax_ppc.set_xticks(f_transf_int_inv([1000, 10000, 100000]))
elif measure == R_measure:
    # ax_ppc.set_xticks([-2.8483186093823685, 0.6190576894032882, 4.086433988188944])
    ax_ppc.set_xticks(f_transf_int_inv([0.1, 1.0]))


ax1.set_xlabel(r'$\mathregular{Pr}(y_i^{\mathregular{rep}} \leq y_i|y_{-i})$')
ax1.set_ylabel(r'$p(\mathregular{Pr}(y_i^{\mathregular{rep}} \leq y_i|y_{-i}))$')

utl.save_plot(plot_settings, f"{__file__[:-3]}_ppc_loopit", measure=args.measure)

az.loo(idata) # normal likelihood on log scale


## correlation with m_ar

m_data = data[['kin', 'm_ar']].drop_duplicates()

cmap = cm.get_cmap('viridis', 8)
cmapn = matplotlib.colors.Normalize(vmin=min(kins), vmax=max(kins))

fig0, ax0 = plt.subplots(figsize=plot_settings["panel_size"])

x = []
y = []

for i, row in m_data.iterrows():
    measure_pred = f_transf_int(trc['b0_intercept']
                                + row['m_ar']*trc['m_ar'])
    
    _x = row['m_ar']
    _y = np.median(measure_pred)
    
    values = az.hdi(measure_pred, hdi_prob=0.95, multimodal=False)

    ax0.plot([_x]*len(values),
             values,
             lw=2,
             color=cmap(cmapn(row['kin'])),
             #solid_capstyle="butt",
             label=row['kin'],
    )
    ax0.plot(_x,
             _y,
             'o',
             color="white",
             mec=cmap(cmapn(row['kin']))
             #solid_capstyle="butt",
    )

ax0.grid(axis = 'both', color='0.9', linestyle='-', linewidth=1)

fig0.colorbar(cm.ScalarMappable(norm=cmapn, cmap=cmap), ax=ax0,
              label="$k_{in}$")

ax0.set_ylabel(measure_name_short)
ax0.set_xlabel('$m_{ar}$')

utl.make_plot_pretty(ax0)


utl.save_plot(plot_settings, f"{__file__[:-3]}_by_m_ar", measure=args.measure)
