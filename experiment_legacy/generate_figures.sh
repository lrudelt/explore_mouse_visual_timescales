
# figure 2

# echo fig 2b
# python3 allen_single_neurons_stimulus.py

# echo fig 2c
# python3 allen_single_neurons.py tau_C

# echo fig 2d
# python3 allen_single_neurons.py R_tot

# this
# echo fig 2a
python3 allen_grouped.py tau_C

# echo fig 2b
python3 allen_grouped.py tau_R

# echo fig 2c
python3 allen_grouped.py R_tot


# echo fig 2e sp
python3 allen_grouped.py tau_C spontaneous

# echo fig 2f sp
python3 allen_grouped.py tau_R spontaneous

# echo fig 2g sp
python3 allen_grouped.py R_tot spontaneous


# echo fig 2e bo
python3 allen_grouped.py tau_C --bo

# echo fig 2f bo
python3 allen_grouped.py tau_R --bo

# echo fig 2g bo
python3 allen_grouped.py R_tot --bo

# this
# echo fig 3a
python3 allen_hierarchy.py tau_C

# echo fig 3b
python3 allen_hierarchy.py tau_R

# echo fig 3c
python3 allen_hierarchy.py R_tot


# echo fig 2h bo
python3 allen_hierarchy.py tau_C --bo

# echo fig 2i bo
python3 allen_hierarchy.py tau_R --bo

# echo fig 2j bo
python3 allen_hierarchy.py R_tot --bo

 figure 11

# echo fig 11a
python3 allen_hierarchy.py tau_C spontaneous

# echo fig 11b
python3 allen_hierarchy.py tau_R spontaneous

# echo fig 11c
python3 allen_hierarchy.py R_tot spontaneous

# echo hierarchy plot with firing rate
python3 allen_hierarchy.py firing_rate

# echo hierarchy plot with median ISI
python3 allen_hierarchy.py median_ISI

# echo hierarchy plot with CV
python3 allen_hierarchy.py CV

# figure 3

# echo fig 3a
python3 allen_violin.py tau_C

# echo fig 3b
python3 allen_violin.py tau_R

# echo fig 3c
python3 allen_violin.py R_tot


# figure 12

# echo fig 12
python3 allen_T0_hierarchy.py tau_C

# echo fig 12 sp
python3 allen_T0_hierarchy.py tau_C spontaneous


# figure 13

# echo fig 13
python3 allen_T0_hierarchy.py tau_C --Tmax500

# echo fig 13 sp
python3 allen_T0_hierarchy.py tau_C spontaneous --Tmax500


# figure 14

# echo fig 14
python3 allen_T0_hierarchy.py tau_R

# echo fig 14 sp
python3 allen_T0_hierarchy.py tau_R spontaneous

# Hierarchical bayesian analysis

# fc natural movie

# model comparison and predictive checks
# python3 allen_hierarchical_bayes_model_comparison.py tau_C

python3 allen_hierarchical_bayes_model_comparison.py tau_R

python3 allen_hierarchical_bayes_model_comparison.py R_tot

# posteriors hierarchy score model
# python3 allen_hierarchical_bayes_hierarchy_score.py tau_C

python3 allen_hierarchical_bayes_hierarchy_score.py tau_R

# python3 allen_hierarchical_bayes_hierarchy_score.py R_tot

# posteriors structure group model
# python3 allen_hierarchical_bayes_struct
# fc natural movie

# model comparison and predictive checks ure_groups.py tau_C

python3 allen_hierarchical_bayes_structure_groups.py tau_R

# python3 allen_hierarchical_bayes_structure_groups.py R_tot


# fc spontaneous

# model comparison and predictive checks
# python3 allen_hierarchical_bayes_model_comparison.py tau_C spontaneous

python3 allen_hierarchical_bayes_model_comparison.py tau_R spontaneous

python3 allen_hierarchical_bayes_model_comparison.py R_tot spontaneous

# posteriors hierarchy score model
# python3 allen_hierarchical_bayes_hierarchy_score.py tau_C spontaneous

python3 allen_hierarchical_bayes_hierarchy_score.py tau_R spontaneous

# python3 allen_hierarchical_bayes_hierarchy_score.py R_tot spontaneous

# posteriors structure group model
# python3 allen_hierarchical_bayes_structure_groups.py tau_C spontaneous

python3 allen_hierarchical_bayes_structure_groups.py tau_R spontaneous

# python3 allen_hierarchical_bayes_structure_groups.py R_tot spontaneous


# bo natural movie

# model comparison and predictive checks
# python3 allen_hierarchical_bayes_model_comparison.py tau_C --bo

python3 allen_hierarchical_bayes_model_comparison.py tau_R --bo

python3 allen_hierarchical_bayes_model_comparison.py R_tot --bo

# posteriors hierarchy score model
# python3 allen_hierarchical_bayes_hierarchy_score.py tau_C --bo

python3 allen_hierarchical_bayes_hierarchy_score.py tau_R --bo

# python3 allen_hierarchical_bayes_hierarchy_score.py R_tot --bo

# posteriors structure group model
# python3 allen_hierarchical_bayes_structure_groups.py tau_C --bo

python3 allen_hierarchical_bayes_structure_groups.py tau_R --bo

# python3 allen_hierarchical_bayes_structure_groups.py R_tot --bo

python3 allen_hierarchical_bayes_model_comparison.py tau_C 

python3 allen_hierarchical_bayes_model_comparison.py tau_R 

python3 allen_hierarchical_bayes_model_comparison.py R_tot 

python3 allen_hierarchical_bayes_model_comparison.py tau_C --bo

python3 allen_hierarchical_bayes_model_comparison.py tau_R --bo

python3 allen_hierarchical_bayes_model_comparison.py R_tot --bo

python3 allen_hierarchical_bayes_model_comparison.py tau_C spontaneous

python3 allen_hierarchical_bayes_model_comparison.py tau_R spontaneous

python3 allen_hierarchical_bayes_model_comparison.py R_tot spontaneous