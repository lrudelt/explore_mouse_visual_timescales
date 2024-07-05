import os
import shutil

import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

data_directory = os.path.abspath("/path/to/repo/experiment_analysis/dat/")
manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

print('get_session_table()')
sessions = cache.get_session_table()

for session_id, row in sessions.iterrows():
    print('downloading data for session {}'.format(session_id))

    truncated_file = True
    directory = os.path.join(data_directory + '/session_' + str(session_id))

    while truncated_file:
        session = cache.get_session_data(session_id)
        try:
            print(session.specimen_name)
            truncated_file = False
        except OSError:
            # shutil.rmtree(directory)
            print(" Truncated spikes file, re-downloading")

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

