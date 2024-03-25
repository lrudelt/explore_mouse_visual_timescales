# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2023-03-07 16:39:01
# @Last Modified: 2023-03-07 16:57:55
# ------------------------------------------------------------------------------ #
# Wrapper to call hdestimator from python to avoid the piping.
# I needed to modify the hde constructor a bit.
# Thus, until it is merged, do not modify the `hde_path`, below.
# ------------------------------------------------------------------------------ #

import sys
import h5py
import tempfile
import os
import pandas as pd
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
)
log = logging.getLogger(__name__)
log.setLevel("DEBUG")


def hde(spiketimes, hde_path="/data.nst/pspitzner/hdestimator", cli_args=dict()):
    if not hde_path in sys.path:
        sys.path.insert(0, hde_path)

    # there is an estimate.py in the hdestimator folder
    from estimate import main

    unique_id = os.uname()[1] + "_" + str(os.getpid()) + "_" + os.urandom(6).hex()
    analysis_dir = tempfile.mkdtemp(prefix=f"hde_analysis_{unique_id}")

    # copy the cli_args, so we can add our own
    cli_args = cli_args.copy()
    cli_args.setdefault(
        "settings-file", os.path.abspath(__file__ + "/../hdestimator_settings.yaml")
    )

    # the following are required for my hack, we cannot overwrite them
    cli_args["analysis_dir"] = analysis_dir
    cli_args["hdf5-dataset"] = "/spiketimes"
    # parsing labels turns out tricky. not always working. also special characters might
    # get stripped. hardcode our own, and check below.
    cli_args["label"] = f"pid_{str(os.getpid())}"

    with tempfile.NamedTemporaryFile(
        prefix=f"dummy_spikefile_{unique_id}",
        delete=True,
        suffix=".h5",
    ) as spikefile:
        with h5py.File(spikefile.name, "w") as f:
            f.create_dataset("/spiketimes", data=spiketimes)

            # not all arguments are supported, e.g. persistence wont work
            # (as its does not have a value)

            arguments = [f"{spikefile.name}"]
            # wihtout perstistent, directory paths are different
            arguments.append("--persistent")
            for k, v in cli_args.items():
                if k.startswith("-"):
                    arguments.append(k)
                elif len(k) == 1:
                    arguments.append(f"-{k}")
                else:
                    arguments.append(f"--{k}")
                arguments.append(v)

            main(arguments)

            # read back the results
            df = pd.read_csv(cli_args["analysis_dir"] + "/ANALYSIS0000/statistics.csv")

            # the dataframe should have only one row where label column is our
            # unique_id
            if len(df) != 1:
                log.error(f'hde returned more than one row, labels: {df["label"].values}')
                log.error(f"this is pid {os.getpid()}, will only return first row")

            res = dict()

            for c in df.columns:
                if c == "label":
                    continue
                if c == "#analysis_num":
                    continue
                res[c] = df[c].values[0]

            try:
                # remove directory and subdirectories
                os.system(f"rm -rf {cli_args['analysis_dir']}")
            except:
                pass

            return res
