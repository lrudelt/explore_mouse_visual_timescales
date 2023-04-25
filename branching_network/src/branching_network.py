# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2023-02-07 11:38:56
# @Last Modified: 2023-03-08 19:13:52
# ------------------------------------------------------------------------------ #

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import logging
import humanize
import zarr
import os
import pyfftw
from tqdm import tqdm
from functools import partialmethod
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg

# we do to 1 / (1 + e^massive_number) = 0.0, numpy complains about this.
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
)
log = logging.getLogger("branching_network")
log.setLevel("DEBUG")


# the @dataclass decorator automatically creates __init__
# with the same arguments as the class attributes.
# we later save them to our zarr files.
@dataclass(kw_only=True)
class BranchingNetwork:
    """
    Creates an instance of a branching network with N neurons.
    Input needs to be added via the `generate_input()` method.

    # Parameters
    N: int
        Number of neurons in the network.
    k: int
        Number of incoming connections per neuron.
    m: float
        Branching parameter.
    input_type: str
        Type of input to the network. Must be one of 'constant', 'OU' or 'scalefree'.
    input_tau: float, required for input_type='OU'
        Timescale of external input.
    input_alpha: float, required for input_type='scalefree'
        Exponent of power-law correlations f^-\alpha.
    rep: int
        Simulation id, used for seed.
    rate: float
        Target rate, Hz per neuron. Only matched in the limit t -> inf.
    sigma: float
        Scale of sensitivity to fluctuations of the input.
    tau_gamma: float
        Timescale of homeostatic regulation, in seconds.
    duration_equil: float
        Calibration period before recording starts.
    duration_record: float
        Simulation duration in seconds, default 20 min.
    dt: float
        Simulation timestep in seconds, default 1 ms.
    N_to_record: int
        Number of neurons to save spike times for.
    path_for_neuron_spiketimes: str
        Path to save neuron spiketimes to. Uses zarr directory store.
        e.g. `../dat/bn_test.zarr`
    meta_note: str
        A string to add, as metadata.
    """

    N: int
    k: int
    m: float

    input_type: str
    input_tau: float = 30.0 / 1000.0
    input_alpha: float = 1.0

    rep: int = 0
    rate: float = 3.5
    sigma: float = 1e-2
    tau_gamma: float = 60.0

    duration_equil: float = 20 * 60
    duration_record: float = 20 * 60
    dt: float = 5.0 / 1000

    N_to_record: int = None
    path_for_neuron_spiketimes: str = None
    meta_note: str = f"created on {datetime.now().strftime('%Y-%m-%d')}"

    def __post_init__(self):
        """
        Our custom init, called automatically after __init__
        Here we build the model
        """
        self.num_timesteps_equil = int(self.duration_equil / self.dt)
        self.num_timesteps_record = int(self.duration_record / self.dt)
        self.num_timesteps = self.num_timesteps_equil + self.num_timesteps_record
        if self.N_to_record is None:
            if self.N > 1000:
                log.warning(
                    f"Recording > 1000 neurons will be slow. Consider setting"
                    f" `N_to_record`."
                )
            self.N_to_record = self.N

        # homeostatic regulation, number of active neurons / N, in each timestep
        self.target_activity = self.rate * self.dt

        self._check_input_args()
        log.debug(
            "\n\t".join(
                ["Branching network with following parameters:"]
                + [f"{k:<27} = {v}" for k, v in self.__dict__.items()]
            )
        )

        # variables for current state
        self.current_timestep = 0
        self.state = np.zeros(shape=self.N, dtype=bool)
        self.gamma = 0
        # input to the network
        self._x_t = None

        # keep track of the total number of spikes for each recorded neuron
        self.spikes_per_neuron = np.zeros(shape=self.N_to_record, dtype="int32")

        np.random.seed(42 + self.rep)

        # ER topology
        # Generate adjacency matrix, random sparse with density k_in / num_nodes
        self.adjacency_matrix = scipy.sparse.random(
            self.N,
            self.N,
            density=self.k / (self.N - 1),
            # lil format is fast for initial creation
            format="lil",
            data_rvs=None,  # use `data_rvs = np.ones` to get homogeneous weights
        )
        # remove self-connections
        self.adjacency_matrix.setdiag(0)

        # normalize each column so that it sums to m
        norm = self.adjacency_matrix.sum(axis=1)
        self.adjacency_matrix = self.adjacency_matrix.multiply(self.m / norm)

        # csr is fast for matrix multiplication
        self.adjacency_matrix = self.adjacency_matrix.tocsr()

        # sanity checks
        log.debug(
            "measured k without self-coupling:"
            f" {np.sum(self.adjacency_matrix > 0)/self.N}"
        )
        log.debug(
            "m measured from adjacency matrix:"
            f" {self.adjacency_matrix.sum(axis=0).mean()}"
        )

        # initialize state to what we know analytically
        # this shortens the required equilibration time
        a = self.target_activity
        self.state = np.random.uniform(size=self.N) < a
        # for constant input
        self.gamma = np.log((1 - self.m * a) / (1 - a) - 1)

        # initalize input here. constant and OU can be calculated in-place
        if self.input_type == "constant":
            pass  # no init required
        elif self.input_type == "OU":
            # empirical estimate of stationary solution for given parameter sigma
            if self.sigma == 1e-3:
                self.gamma = -6120
            elif self.sigma == 1e-2:
                self.gamma = -612
            self._x = np.random.normal(0.0, 1.0, self.N)
        elif self.input_type == "scalefree":
            # this is more involved. sets the _x_t attribute
            self._precompute_scalefree_input()

    def run(self, callback=lambda: None):
        """
        Run the network for the number of timesteps specified in the constructor.
        (Input generation depends on num_timesteps, thus we specify them already there.)

        # Parameters
        callback: function
            called after each timestep (without any arguments)
        """

        self.current_timestep = 0

        # safe some time series for the equilibriation phase
        self._equil_ts_rate = np.zeros(shape=self.num_timesteps_equil)
        self._equil_ts_gamma = np.zeros(shape=self.num_timesteps_equil)
        self._equil_ts_activity = np.zeros(shape=self.num_timesteps_equil)

        # safe time series for recordings
        self.ts_rate = np.zeros(shape=self.num_timesteps_record)
        self.ts_gamma = np.zeros(shape=self.num_timesteps_record)
        self.ts_activity = np.zeros(shape=self.num_timesteps_record)

        log.info("Equilibrating")
        for step in tqdm(range(self.num_timesteps_equil), desc="Equilibrating"):
            self.step()

            act = np.sum(self.state) / self.N
            self._equil_ts_activity[step] = act
            self._equil_ts_rate[step] = act / self.dt
            self._equil_ts_gamma[step] = self.gamma

            callback()


        # we dont write to disk after each step, but flush occasionally
        self._init_chunked_spike_store()

        log.info("Recording")
        for step in tqdm(range(self.num_timesteps_record), desc="Recording"):
            self.step()

            act = np.sum(self.state) / self.N
            self.ts_activity[step] = act
            self.ts_rate[step] = act / self.dt
            self.ts_gamma[step] = self.gamma

            # update our record of how often each recorded neuron spiked.
            self.spikes_per_neuron[:] += self.state[0 : self.N_to_record]
            self._cache_spikes()

            callback()

        self._flush_spike_cache()

        log.info(f"simulation run done!")
        log.debug(
            f"measured mean rate over {step+1} steps: {np.mean(self.ts_rate):0.3f} Hz"
        )

        self._save_m_from_autoregressive()

    def step(self):
        """
        One timestep of the network. Uses dot products to compute the state update.
        Check the equation in the SM for details.
        """

        past_state = self.state
        x = self.get_current_input()

        p_ext = 1 / (1 + np.exp(-x / self.sigma - self.gamma))

        # the clipping is just for safety, should not be needed since we normalized
        # columns to m during init. also implements rectified linar
        p_rec = np.clip(self.adjacency_matrix.dot(past_state), 0, 1)

        # total activation probability accounting for coalescence,
        # DOI: 10.1103/PhysRevE.101.022301
        # 1 - p_no_ext * p_no_rec
        p_act = 1.0 - (1.0 - p_ext) * (1.0 - p_rec)

        self.state = np.random.uniform(size=self.N) < p_act

        # update gamma, homeostatic regulation which is multiplicative to have a
        # relative change to the scale of gamma
        self.gamma += np.fabs(self.gamma) * (
            (self.target_activity - np.sum(self.state) / self.N)
            * self.dt / self.tau_gamma
        )

        self.current_timestep += 1

    def get_current_input(self):
        """
        Get the input for the current timestep, respecting the input_type.
        """
        if self.input_type == "constant":
            return np.zeros(self.N)

        if self.input_type == "OU":
            # update x for next timestep
            self._x += -self.dt / self.input_tau * self._x + np.random.normal(
                0.0, 1.0, self.N
            )
            return self._x

        if self.input_type == "scalefree":
            return self._x_t[:, self.current_timestep]

    def _check_input_args(self):
        error_msg = "\n".join(
            [
                "Invalid input arguments!",
                "`input_type` must be 'constant', 'OU' or 'scalefree'.",
                "For `OU`, also `input_tau` must be specified.",
                "For `scalefree`, also `input_alpha` must be specified.",
            ]
        )
        if self.input_type not in ["constant", "OU", "scalefree"]:
            raise ValueError(error_msg)

        # overwrite unused attributes
        if self.input_type == "constant":
            self.input_alpha = None
            self.input_tau = None
        elif self.input_type == "OU":
            self.input_alpha = None
        elif self.input_type == "scalefree":
            self.input_tau = None

    def _precompute_scalefree_input(self):
        """
        scalefree input needs to be precompuated using FFTs.

        Thus, generate inputs for the network, once in the beginning.
        We create Unique time series for each node.
        This sets the attribute self._x_t, but does not return anything.
        See Zierenberg et al. 10.1103/PhysRevE.96.062125
        """

        self._check_input_args()
        log.debug(f"Generating scalefree input with alpha {self.input_alpha}")

        self._x_t = np.zeros((self.N, self.num_timesteps))
        log.debug(
            "Memory usage for input x_t is:"
            f" {humanize.naturalsize(self._x_t.nbytes, binary=True)}"
        )

        # DFT 101:
        # df = 1 / (N dt)
        # DFT should return
        #   for odd  N: 0, df, 2 df, ..., (N/2)df, -(N/2)df, ..., -df
        #   for even N: 0, df, 2 df, ..., (+/-) N/2 df, (-N/2+1) df, ..., -df
        freqs = np.fft.fftfreq(self.num_timesteps) / self.dt

        # power-law power spectral density
        psd = np.zeros(len(freqs))
        # avoid first element to prevent divergence
        psd[1:] = np.abs(freqs[1:]) ** (-self.input_alpha)
        psd[0] = psd[1]
        # normalize power spectrum such that variance (correlation at lag 0) is one
        # Wiener Kinchim theorem says that C(0) = 1/N * sum(S(k))
        psd *= self.num_timesteps / psd.sum()

        # iterate over nodes taking only every second, because we get two indep. series
        for n in tqdm(range(0, self.N, 2), desc="generating input for neurons"):
            # generate random numbers from normal dist with mean variance given by psd
            real = np.random.normal(
                size=psd.shape, scale=np.sqrt(psd * self.num_timesteps)
            )
            imag = np.random.normal(
                size=psd.shape, scale=np.sqrt(psd * self.num_timesteps)
            )

            # Fourier transform complex frequency space to complex time domain
            # freqs = np.empty(num_freqs)
            phi_f = pyfftw.empty_aligned(len(freqs), dtype="complex128", n=16)
            phi_f[:] = real + 1j * imag

            # discrete Fourier transform into time domain (phi_t is Gaussian
            # random variable with mean zero, variance 1, in domain (-infty, infty))
            phi_t = pyfftw.interfaces.numpy_fft.ifft(phi_f)

            self._x_t[n] = np.real(phi_t)
            if n + 1 < self.N:
                self._x_t[n + 1] = np.imag(phi_t)

    # ------------------------------------------------------------------------------ #
    # non-model related helper, saving, caching, metadata
    # ------------------------------------------------------------------------------ #

    def _init_chunked_spike_store(self, spikes_per_chunk=1_000_000):
        """
        For large networks and long simulations, the spike timeseries wont fit into
        memory. We create a chunked zarr array.

        Creates two zarr directory stores at `self.path_for_neuron_spiketimes`,
        "/spikes_as_list" and "/spikes_per_neuron".

        we aim for ~ 5mb chunk size, saving as int32 i.e. 4 bytes per entry
        5mb * 1024 * 1024 / 4 ~ 1_300_000
        """

        # create a cache in ram that we flush to disk every now and then
        self.spike_cache_nids = []  # neuron ids
        self.spike_cache_times = []  # spike times
        self.spikes_per_chunk = spikes_per_chunk

        if self.path_for_neuron_spiketimes is None:
            log.info(
                "no path for neuron spiketimes given, all spikes will be kept in memory"
                " (spike_cache_nids, spike_cache_times)"
            )
            return

        # create a zarr array with two columns, one for neuron id, one for spike time
        store = zarr.storage.DirectoryStore(self.path_for_neuron_spiketimes)
        self.spikes_as_list = zarr.open(
            store=store,
            path="/spikes_as_list",
            mode="w",
            shape=(2, 0),
            chunks=(2, self.spikes_per_chunk),
            dtype="int32",
        )
        log.debug(f"saving spikes as list to:\n\t{store.path}")
        info = str(self.spikes_as_list.info).replace("\n", "\n\t")
        log.debug(f"zarr for spikes as list:\n\t{info}")

        # second format, making each neuron readable alone.
        # we simply add to the existing store, but in another group and read-optimized
        # chunks.
        self._spikes_by_neuron_next_write_index = np.zeros(
            shape=self.N_to_record, dtype=int
        )
        self.spikes_by_neuron = zarr.open(
            store=self.spikes_as_list.store,
            path="/spikes_by_neuron/",
            mode="w",
            shape=(self.N_to_record, 0),
            chunks=(1, spikes_per_chunk),
            dtype="int32",
        )

        self._save_metadata()

    def _save_metadata(self):
        """
        save extra details and metadata to the zarr stores.
        the stores are created in init_chunked_spike_store.
        """

        self.spikes_as_list.attrs[
            "dataformat_details"
        ] = """
        Neuron spiketimes are saved as integer with the first column
        being the neuron id and the second id the time step.
        To get spiketimes in seconds, multiply the second column with
        the timestep (`loaded_array.attrs['dt']`).
        """

        self.spikes_by_neuron.attrs[
            "dataformat_details"
        ] = """
        Spiketimes saved as 2d padded array of integer with shape
        (N, max_spikes_per_neuron). Unused entries are padded with -1.
        First index is neuron id (outer list)
        Second index holds the time step.
        To get spiketimes in seconds, multiply with the timestep
        (`loaded_array.attrs['dt']`).
        """

        for dset in [self.spikes_as_list, self.spikes_by_neuron]:
            try:
                # git repo etc.
                md = get_extended_metadata(prefix="meta_")
                for k, v in md.items():
                    dset.attrs[k] = v
            except Exception as e:
                log.debug(f"failed to save extended metadata: {e}")

            # all model details, since we set them as attributes
            save_object_attributes(obj=self, dset=dset)

    def _cache_spikes(self, flush=False):
        """
        Append spikes from current state to the cached spike timeseries and flush
        to disk if necessary.

        set flush=True to force a flush to disk.
        """
        nids = np.where(self.state)[0]
        # For now, lets simply keep the first neurons, connectivity is random anyway.
        nids = nids[nids < self.N_to_record]
        times = np.ones_like(nids) * (self.current_timestep - self.num_timesteps_equil)

        self.spike_cache_nids.extend(nids)
        self.spike_cache_times.extend(times)

        if self.path_for_neuron_spiketimes is None:
            return

        if flush or len(self.spike_cache_nids) > self.spikes_per_chunk:
            self._flush_spike_cache()

    def _flush_spike_cache(self):
        log.debug(
            "flushing spike cache to disk, "
            f" {self.current_timestep/self.num_timesteps*100:.1f} % of simulation"
        )

        # in the as-list format
        self.spikes_as_list.append(
            np.array([self.spike_cache_nids, self.spike_cache_times]),
            axis=1,
        )

        # in the by-neuron format
        # for large N_to_record, this will get slow.
        # resize store on disk to the max number of spikes per neuron
        self.spikes_by_neuron.resize(self.N_to_record, np.max(self.spikes_per_neuron))

        # we need the list to array cast, otherwise we dont find anything with np.where
        spike_cache_nids = np.array(self.spike_cache_nids)
        spike_cache_times = np.array(self.spike_cache_times)

        for nid in range(self.N_to_record):
            sids = np.where(spike_cache_nids == nid)
            spks_to_save = spike_cache_times[sids]
            # spks_to_save has shape (num_spikes,)
            beg_idx = self._spikes_by_neuron_next_write_index[nid]
            end_idx = beg_idx + spks_to_save.shape[-1]
            self.spikes_by_neuron[nid, beg_idx:end_idx] = spks_to_save
            # mark the rest of this neurons array as invalid
            self.spikes_by_neuron[nid, end_idx:] = -1
            self._spikes_by_neuron_next_write_index[nid] = end_idx


        # reset cache
        self.spike_cache_nids = []
        self.spike_cache_times = []

        try:
            # we collaborate on this, make sure group permissions are okay
            os.system(f"chmod -R g+rwx {self.path_for_neuron_spiketimes}")
        except:
            log.debug("failed to set permissions on files for spike timeseries")


    def _save_m_from_autoregressive(self):
        """
        calculate m with autorgression from time series and save as zarr attribute.
        only call this once at the end of the simulation run.
        """
        if len(self.ts_activity) == 0:
            return
        ar_mod = AutoReg(self.ts_activity, lags=1)
        ar_res = ar_mod.fit()
        m_AR = ar_res.params[1]
        log.debug(
            f"estimated m based on autoregression of activity timeseries: {m_AR:0.3f}"
        )

        try:
            self.spikes_as_list.attrs["m_AR"] = m_AR
        except:
            log.debug("failed to save m_AR to spikes_as_list")
        try:
            self.spikes_by_neuron.attrs["m_AR"] = m_AR
        except:
            log.debug("failed to save m_AR to spikes_by_neuron")


# ------------------------------------------------------------------------------ #
# module level helper
# ------------------------------------------------------------------------------ #
# on the cluster we dont want to see the progress bar
# https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
def disable_tqdm():
    """Disable tqdm progress bar."""
    global tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# disable by default
disable_tqdm()


def enable_tqdm():
    """Enable tqdm progress bar."""
    global tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)


def save_object_attributes(obj, dset):
    """
    trying something new here, hoping its more future proof than saving
    metadata manually.

    saves all attributes of an object as attributes of the zarr group.
    lists and arrays asigned a string representation of their type and length.
    attributes starting with "_" are skipped.

    # Parameters
    obj: object
        object to save attributes from
    dset: zarr group or the like.
        needs to support the dset.attrs["foo"] = "bar" syntax
    """

    for key, val in obj.__dict__.items():
        if key.startswith("_"):
            continue
        rep = None
        # do not save lists, arrays etc as attributes
        try:
            if len(val.shape) == 0:
                # np scalars might have a shape but are just numbers
                raise ValueError
            rep = f"{str(type(val))[8:-2:]} with shape {val.shape}"
        except:
            if isinstance(val, (dict, list, set, tuple)):
                rep = f"{str(type(val))[8:-2:]} with length {len(val)}"
            elif val is None:
                rep = f"None"
            else:
                rep = val

        if rep is not None:
            dset.attrs[key] = rep
        else:
            log.debug(f"Failed to save {key}")


def get_extended_metadata(prefix="meta_"):
    """
    Get a bunch of standard metadata. Usually good to add as model attributes.
    Git details need the git package: pip install gitpython

    # Parameters
    prefix : str
        Prefix to add to the keys of the metadata dictionary. default: "meta_"

    # Returns
    metadata : dict

    # Example
    ```python
    model = BasicHierarchicalModel()
    # add all dict keys as attributes
    model.__dict__.update(get_metadata())
    ```
    """
    p = prefix
    md = dict()
    md[f"{p}hostname"] = os.uname()[1]
    md[f"{p}username"] = os.environ.get("USER", None)

    # check for slurm
    if "SLURM_JOB_ID" in os.environ:
        # https://docs.hpc.shef.ac.uk/en/latest/referenceinfo/scheduler/SLURM/SLURM-environment-variables.html
        md[f"{p}slurm_job_id"] = os.environ.get("SLURM_JOB_ID", None)
        md[f"{p}slurm_job_array_id"] = os.environ.get("SLURM_ARRAY_JOB_ID", None)
        md[f"{p}slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        md[f"{p}slurm_submit_dir"] = os.environ.get("SLURM_SUBMIT_DIR", None)
    if "SGE_TASK_ID" in os.environ:
        # https://docs.oracle.com/cd/E19957-01/820-0699/chp4-21/index.html
        md[f"{p}job_id"] = os.environ.get("JOB_ID", None)
        md[f"{p}sge_task_id"] = os.environ.get("SGE_TASK_ID", None)
        md[f"{p}sge_o_workdir"] = os.environ.get("SGE_O_WORKDIR", None)
        md[f"{p}sge_cwd_path"] = os.environ.get("SGE_CWD_PATH", None)

    import sys

    # conda and python related stuff
    md[f"{p}python_exe"] = sys.executable
    md[f"{p}conda_exe"] = os.environ.get("CONDA_EXE", None)

    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        md[f"{p}git_url"] = repo.remotes.origin.url
        md[f"{p}git_commit"] = repo.head.object.hexsha
        md[f"{p}git_branch"] = repo.active_branch.name
        md[f"{p}git_commit_message"] = repo.head.object.message
    except:
        log.debug("Could not import gitpython. Not saving git detail.")
        md[f"{p}git_url"] = None
        md[f"{p}git_commit"] = None
        md[f"{p}git_branch"] = None
        md[f"{p}git_commit_message"] = None

    return md


# ------------------------------------------------------------------------------ #
# thin wrapper to be able to run simulations from command line
# ------------------------------------------------------------------------------ #


def main():
    """
    This simply calls the constructor and passes comman line arguments as keywords.

    # Example
    ```bash
    python3 ./src/branching_network.py \
        -N=100 -k=10 \
        -input_type="OU" \
        -meta_note="lorem ipsum"
    ```
    """
    import argparse

    parser = argparse.ArgumentParser()
    _, args = parser.parse_known_args()

    log.info(f"Branching Network from command line with args:")
    for arg in args:
        log.info(arg)

    # let useful metadata appear in the log file
    try:
        meta = get_extended_metadata(prefix="")
        log.info(f"Extended metadata for this thread:")
        for k, v in meta.items():
            log.info(f"{k}: {v}")
    except:
        pass

    # in the command line we do not want progress bars (usually run on cluster)
    disable_tqdm()

    # remove leading dashes
    args = [arg.lstrip("--") for arg in args]
    args = [arg.lstrip("-") for arg in args]

    # create a dict of keyword arguments that we can pass to the constructor
    kwargs = {k: v for k, v in [arg.split("=") for arg in args]}

    # try to cast to float, or to boolean if = "True" or "False"
    for k, v in kwargs.items():
        if v == "True" or v == "False":
            kwargs[k] = eval(v)
        elif v == "None":
            kwargs[k] = None
        else:
            # try integers first, then floats
            try:
                kwargs[k] = int(v)
            except ValueError:
                try:
                    kwargs[k] = float(v)
                except ValueError:
                    pass

    log.info(f"passing kwargs to constructor: {kwargs}")

    # we are not super careful with casting here. passing debug=False will not work.
    # use debug=0, and double check for other issues
    bn = BranchingNetwork(**kwargs)
    bn.run()


if __name__ == "__main__":
    main()
