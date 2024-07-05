from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import build_and_execute
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import os
import urllib

rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")

data_directory = os.path.abspath("/path/to/repo/experiment_analysis/dat/")
manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()


def retrieve_link(session_id):

    well_known_files = build_and_execute(
        (
        "criteria=model::WellKnownFile"
        ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
        "[attachable_type$eq'EcephysSession']"
        r"[attachable_id$eq{{session_id}}]"
        ),
        engine=rma_engine.get_rma_tabular,
        session_id=session_id
    )

    return 'http://api.brain-map.org/' + well_known_files['download_link'].iloc[0]

# download_links = [retrieve_link(session_id) for session_id in sessions.index.values]

# _ = [print(link) for link in download_links]

for session_id in sessions.index.values:
# for session_id, row in sessions.iterrows():

    session_dir = os.path.join(data_directory, "session_{}".format(session_id))
    try:
        os.mkdir(session_dir)
    except:
        pass

    link = retrieve_link(session_id)
    print("downloading session {}..".format(session_id))

    urllib.request.urlretrieve(link, "{}/session_{}.nwb".format(session_dir,
                                                                session_id))

