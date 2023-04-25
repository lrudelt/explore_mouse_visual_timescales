# Code repository: _Signatures of hierarchical temporal processing in the mouse visual system_


This repository contains the analysis pipeline and plotting routines for our preprint

[Arxiv]()

```
@misc{rudelt_signatures_2023,
  doi = {},
  url = {},
}
```

## Data

(work in progress)

We analyse data of mouse visual cortex from the [Allen Brain Atlas](https://atlas.brain-map.org/).
The data is accessed using the [Allen SDK](http://alleninstitute.github.io/AllenSDK/install.html).
For convenience, we (will) provide a copy of the preprocessed data that is compatible with
our analysis pipeline on [gin.g-node.org](https://gin.g-node.org/pspitzner/mouse_visual_timescales).
Loading these files requires only minimal dependencies and should be easy to setup using
our `environment.yaml`.



## Analysis

(work in progress)

### Planned:
All analysis run on our preprocessed data and all requirements can be installed
by creating a new conda environment

```
conda env create -f environment.yaml --name mouse_visual_timescales
```

### Current state:
Most of the analysis scripts have not been updated to the new data format.
Plotting works, however, using intermediate results saved in `todo@LR`

## Branching Network

Simulations of the branching network are independent from the experimental anlysis.
Everything is contained in `branching_network`, along with a separate environment file
for dependencies.


## Plotting

### Figure 1

### Figure 2

### Figure 3
All panels are produced by `branching_network/notebooks/bn_plot_measures_vs_a.ipynb`
and `branching_network/notebooks/bn_raster_examples.ipynb`.

### Figure 4
