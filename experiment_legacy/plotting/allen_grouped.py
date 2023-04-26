import argparse
from bitsandbobs.plt import set_size, alpha_to_solid_on_bg

defined_measures = ["tau_C", "tau_R", "R_tot"]
defined_stimuli = ["movie", "spontaneous"]
parser = argparse.ArgumentParser()
parser.add_argument("measure", type=str, help=f"one of {defined_measures}")
parser.add_argument(
    "stimulus",
    type=str,
    help=f"one of {defined_stimuli}, default: movie",
    nargs="?",
    default="movie",
)
parser.add_argument(
    "--bo",
    dest="allen_bo",
    action="store_true",
    help=f"use brain observatory data set for movie stimulus",
)

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
with open("{}/dirs.yaml".format(SCRIPT_DIR), "r") as dir_settings_file:
    dir_settings = yaml.load(dir_settings_file, Loader=yaml.BaseLoader)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from sys import path

path.insert(1, "../../allen_src/")
import analysis_utils as utl

from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors
import seaborn as sns

### settings

if args.allen_bo:
    analysis = "allen_bo"
else:
    analysis = "allen_fc"

center = "median"  # measure of central tendency
T_measure = "tau_R"
R_measure = "R_tot"
C_measure = "tau_two_timescales"

if args.allen_bo:
    stimuli = ["natural_movie_three"]
elif args.stimulus == "movie":
    stimuli = ["natural_movie_one_more_repeats"]
elif args.stimulus == "spontaneous":
    stimuli = ["spontaneous"]
T0 = 0.03  # 30 ms
selected_structures = "cortex+thalamus"

plot_stars = False  # plot stars or p-values for tests

# setup analysis
plot_settings = utl.get_default_plot_settings()
plot_settings["panel_width"] = 2.0
plot_settings["panel_height"] = 2.7
plot_settings["rcparams"]["figure.figsize"] = [
    plot_settings["panel_width"],
    plot_settings["panel_height"],
]
plt.rcParams.update(plot_settings["rcparams"])

stats_dir = dir_settings["stats_dir"]

csv_file_name = "{}/{}_statistics_T0_30.csv".format(stats_dir, analysis)
mre_stats_file_name = "{}/{}_mre_statistics_T0_30_Tmax_10000_two_timescales.csv".format(
    stats_dir, analysis
)

structures = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam", "LGd", "LP"]
structures_map = utl.get_structures_map()
structures_cortex = [
    structure
    for structure in structures
    if structures_map[structure]["parent_structure"] == "cortex"
]

### import data

data = utl.get_analysis_data(
    csv_file_name, analysis, mre_stats_file_name=mre_stats_file_name
)

# make sure data as expected
if selected_structures == "cortex+thalamus":
    _structures = structures
elif selected_structures == "cortex":
    _structures = structures_cortex
else:
    print("unknown structures selected")
    exit()

data = data[utl.df_filter(data, structures=_structures, stimuli=stimuli, T0=T0, tmin=T0)]
try:
    num_neurons = utl.get_expected_number_of_neurons(
        analysis, selected_structures, stimuli
    )
    assert np.isclose(len(data), num_neurons, atol=100)
except:
    print(f"number of neurons is {len(data)}, expected {num_neurons}")
    exit()

# plot

fig0, ax = plt.subplots(
    figsize=(plot_settings["panel_width"], plot_settings["panel_height"])
)
fig0.subplots_adjust(right=0.98, left=0.1, bottom=0.1)

if args.allen_bo:
    if args.measure == "tau_C":
        lims = [0.14, 0.75]
        measure = C_measure
    elif args.measure == "tau_R":
        lims = [0.017, 0.106]
        measure = T_measure
    elif args.measure == "R_tot":
        lims = [0.047, 0.115]
        measure = R_measure
elif args.stimulus == "movie":
    if args.measure == "tau_C":
        lims = [0.16, 0.66]
        measure = C_measure
    elif args.measure == "tau_R":
        lims = [0.016, 0.086]
        measure = T_measure
    elif args.measure == "R_tot":
        lims = [0.047, 0.115]
        measure = R_measure
elif args.stimulus == "spontaneous":
    if args.measure == "tau_C":
        lims = [0.18, 0.62]
        measure = C_measure
    elif args.measure == "tau_R":
        lims = [0.019, 0.089]
        measure = T_measure
    elif args.measure == "R_tot":
        lims = [0.043, 0.096]
        measure = R_measure


structure_group_names = ["thalamus", "V1", "higher\ncortical"]
structure_group_components = [
    ["LGd", "LP"],
    ["VISp"],
    ["VISl", "VISrl", "VISal", "VISpm", "VISam"],
]

x_pos = [-0.2, 0.75, 2]

# lets say 0.5 between groups
gdx = 0.5
# lets say 0.25 between in-group markers
mdx = 0.25
# padding around group edges
pdx = 0.1

x_pos = {}

_ref = pdx
x_pos["thalamus"] = {
    "groupleft": _ref - pdx,
    "groupright": _ref + 2 * mdx + pdx,
    "groupmarker": _ref + 2 * mdx,
    "LGd": _ref,
    "LP": _ref + 1 * mdx,
}
_ref = x_pos["thalamus"]["groupright"] + gdx + pdx
x_pos["V1"] = {
    "groupleft": _ref - pdx,
    "groupright": _ref + 1.5 * mdx + pdx,
    # "groupmarker": _ref + 0.5 * mdx,
    "VISp": _ref + 0.75 * mdx,
}
_ref = x_pos["V1"]["groupright"] + gdx + pdx
x_pos["higher\ncortical"] = {
    "groupleft": _ref - pdx,
    "groupmarker": _ref,
    "groupright": _ref + 5 * mdx + pdx,
    "VISl": _ref + 1 * mdx,
    "VISrl": _ref + 2 * mdx,
    "VISal": _ref + 3 * mdx,
    "VISpm": _ref + 4 * mdx,
    "VISam": _ref + 5 * mdx,
}

_xticks = []
_xlabels = []

selection = utl.get_data_filter(data, measure)
data = data[selection]

bp_data = []
mwu_data = []


for (
    structure_group_name,
    _structure_group_components,
) in zip(structure_group_names, structure_group_components):

    if structure_group_name == "thalamus":
        _alpha = 0.4
        _black = "0.6"
    else:
        _alpha = 1.0
        _black = "0.3"

    # black group levle markers
    _y = utl.get_center(
        data[utl.df_filter(data, structures=_structure_group_components)][measure].values,
        center,
    )
    _y_err = (
        utl.get_sd(
            data[utl.df_filter(data, structures=_structure_group_components)][
                measure
            ].values,
            center,
        ),
    )
    _y_CI_low, _y_CI_high = utl.get_CI_median(
        data[utl.df_filter(data, structures=_structure_group_components)][measure].values
    )

    if "groupmarker" in x_pos[structure_group_name]:
        _x = x_pos[structure_group_name]["groupmarker"]
        ax.plot(
            [_x, _x],
            [_y_CI_low, _y_CI_high],
            lw=2,
            color=_black,
        )
        ax.plot([_x], [_y], "o", color=_black, ms=3.5)

    _left = x_pos[structure_group_name]["groupleft"]
    _right = x_pos[structure_group_name]["groupright"]
    ax.plot(
        [_left, _right],
        [_y, _y],
        lw=2,
        color=_black,
        solid_capstyle="round",
    )

    mwu_data += [_y + _y_err]

    for structure in _structure_group_components:
        _y = utl.get_center(
            data[utl.df_filter(data, structures=structure)][measure].values, center
        )
        _y_err = utl.get_sd(
            data[utl.df_filter(data, structures=structure)][measure].values, center
        )
        _y_CI_low, _y_CI_high = utl.get_CI_median(
            data[utl.df_filter(data, structures=structure)][measure].values
        )
        __x = x_pos[structure_group_name][structure]
        ax.plot(
            [__x, __x],
            # [_y - _y_err, _y + _y_err],
            [_y_CI_low, _y_CI_high],
            lw=2,
            # alpha=.4,
            color=alpha_to_solid_on_bg(structures_map[structure]["color"], _alpha),
        )
        ax.plot(
            __x,
            _y,
            "o",
            color="white",
            mec=alpha_to_solid_on_bg(structures_map[structure]["color"], _alpha),
        )

        _xticks += [__x]
        _xlabels += [structures_map[structure]["name"]]

    bp_data += [
        data[utl.df_filter(data, structures=_structure_group_components)][measure].values
    ]

# have a dict mapping structure names to x positions
x_labels = dict()

for i, j, y_off_f in zip([0, 1, 0], [1, 2, 2], [0, 1, 2]):
    MW_u, MW_p = st.mannwhitneyu(bp_data[i], bp_data[j], alternative="two-sided")

    i_pos = x_pos[structure_group_names[i]]
    j_pos = x_pos[structure_group_names[j]]
    x1 = i_pos["groupright"] - (i_pos["groupright"] - i_pos["groupleft"]) / 2
    x2 = j_pos["groupright"] - (j_pos["groupright"] - j_pos["groupleft"]) / 2

    # reuse to set label positions
    x_labels[structure_group_names[i]] = x1
    x_labels[structure_group_names[j]] = x2

    if MW_p < 0.05:
        w = 0.05 * (lims[1] - lims[0])

        y = max([max(d) for d in mwu_data]) + 1.5 * w

        y_off = y_off_f * w * 2
        y += y_off

        if plot_stars:
            if MW_p < 0.001:
                num_stars = 3
            elif MW_p < 0.01:
                num_stars = 2
            else:
                num_stars = 1
            text = "*" * num_stars
            text_y_pos = y + 2 * w
        else:
            if MW_p < 0.0005:
                # text = '$p < 10^{-3}$'
                text = "$p < 0.001$"
            else:
                text = "$p$ = {:.3f}\n".format(MW_p)
            text_y_pos = y + 2.5 * w

        ax.plot([x1, x2], [y + w, y + w], lw=1.5, c="k")
        ax.text((x1 + x2) / 2, text_y_pos, text, ha="center", va="top", color="k")


# thalamus takes a lot of space, move V1 label a bit to the right
x_labels["V1"] += 0.15
ax.set_xticks(list(x_labels.values()))
ax.set_xticklabels(list(x_labels.keys()))
ax.set_ylim(lims)

utl.format_ax(ax, measure, "y", set_ticks=False, tight_layout=False)
utl.make_plot_pretty(ax)

# remove x axis ticks and line
ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=True)
ax.spines["bottom"].set_visible(False)

# set_size(ax=ax, w = plot_settings['panel_width'], h=plot_settings['panel_height'])

utl.save_plot(
    plot_settings,
    f"allen_grouped",
    allen_bo=args.allen_bo,
    stimulus=args.stimulus,
    measure=args.measure,
)
