# %%
import os
import shutil

import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

data_directory = os.path.abspath("/path/to/repo/experiment_analysis/dat/")
manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

presence_ratio_minimum = 0.9
amplitude_cutoff_maximum = 0.01
isi_violations_maximum = 0.5

print('get_session_table()')
sessions = cache.get_session_table()

#%%
analysis_metrics_brain_observatory = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1', amplitude_cutoff_maximum = amplitude_cutoff_maximum,
                                                          presence_ratio_minimum = presence_ratio_minimum,
                                                          isi_violations_maximum = isi_violations_maximum)

analysis_metrics_functional_connectivity = cache.get_unit_analysis_metrics_by_session_type('functional_connectivity', amplitude_cutoff_maximum = amplitude_cutoff_maximum,
                                                          presence_ratio_minimum = presence_ratio_minimum,
                                                          isi_violations_maximum = isi_violations_maximum)

 #%% Save metric data

analysis_metrics_brain_observatory.to_csv("brain_observatory_unit_metrics_filtered.csv")
    # don't download lfp data

    # for probe_id, probe in session.probes.iterrows():

    #     print(' ' + probe.description)
    #     truncated_lfp = True

    #     while truncated_lfp:
    #         try:
    #             lfp = session.get_lfp(probe_id)
    #             truncated_lfp = False
    #         except OSError:
    #             fname = directory + '/probe_' + str(probe_id) + '_lfp.nwb'
    #             os.remove(fname)
    #             print("  Truncated LFP file, re-downloading")
    #         except ValueError:
    #             print("  LFP file not found.")
    #             truncated_lfp = False


# %%
