import pymc as pm
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class ModelBase(pm.Model):
    """BaseClass, here we do some preprocessing of the data that is needed for all models.

    Args:
        pm (_type_): _description_
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_data(self, df: pd.DataFrame):

        df = df.copy()

        # Log transforms
        cols_to_trafo = ["firing_rate", "R_tot", "tau_R", "tau_double"]
        num_rows_before = len(df)
        for c in cols_to_trafo:
            log_values = np.log(df[c])
            #  pd.dropna does not drop inf/-inf, so we have to cast them.
            log_values = np.where(np.isinf(log_values), np.nan, log_values)
            df[f"log_{c}"] = log_values

        df = df.dropna(subset=[f"log_{c}" for c in cols_to_trafo])
        num_rows_after = len(df)
        log.info(f"dropped {num_rows_before - num_rows_after} rows due to inf/nans")

        # z-transfrom
        # -----------
        # and mean-centered & standardized according to Gelman: 1 / (2 * sd)
        # Divide by 2 standard deviations in order to put the variance of a
        # normally distributed variable nearer to the variance range of a
        # binary variable. See
        # http://www.stat.columbia.edu/~gelman/research/published/standardizing7.pdf
        # for more info.

        def z_trafo(col):
            return (col - col.mean()) / col.std()

        def gelman_mean(col):
            return (col - col.mean()) / 2.0 / col.std()

        def gelman_min(col):
            return (col - col.min()) / 2.0 / col.std()

        for c in ["R_tot", "tau_R", "tau_double"]:
            df[f"z_log_{c}"] = z_trafo(df[f"log_{c}"])

        for c in ["firing_rate"]:
            df[f"z_log_{c}"] = gelman_mean(df[f"log_{c}"])

        for c in ["hierarchy_score"]:
            df[f"z_{c}"] = gelman_min(df[f"{c}"])

        self.df = df
        return self.df


class LinearMedianModel(ModelBase):
    def __init__(self, df, measure, noncentered=True, name=""):

        # this copies df, prepares and assigns self.df
        df = self.prepare_data(df)

        # to model session-level details, we need a lookup row -> session_idx
        session_idx, sessions = pd.factorize(df["session"])
        df["session_idx"] = session_idx
        num_sessions = len(sessions)

        # it helps readability of arviz to define dimensionality via coords:
        # https://www.pymc.io/projects/docs/en/v5.10.2/learn/core_notebooks/dimensionality.html#dims
        # this enables e.g. `idata.sel(session=0)`
        coords = {
            # we could use the real sessions of the data, but this makes it harder to index.
            "session": np.unique(session_idx),
            "datapoint": np.arange(len(df)),
        }

        super().__init__(name=name, model=None, coords=coords)

        # hyperpriors
        mu_intercept = pm.Normal("mu_intercept", mu=0.0, sigma=1.0)
        sigma_intercept = pm.HalfCauchy("sigma_intercept", beta=1.0)

        mu_slope = pm.Normal("mu_slope", mu=0.0, sigma=1.0)
        sigma_slope = pm.HalfCauchy("sigma_slope", beta=1.0)

        if noncentered is False:
            session_intercept = pm.Normal(
                "session_intercept",
                mu=mu_intercept,
                sigma=sigma_intercept,
                shape=num_sessions,
                dims="session",
            )

            session_slope = pm.Normal(
                "session_slope",
                mu=mu_slope,
                sigma=sigma_slope,
                shape=num_sessions,
                dims="session",
            )
        else:
            # avoid funnel of hell
            # https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
            session_intercept_scaled = pm.Normal(
                "session_intercept_scaled",
                mu=0.0,
                sigma=1.0,
                shape=num_sessions,
                dims="session",
            )
            session_intercept = pm.Deterministic(
                "session_intercept",
                mu_intercept + session_intercept_scaled * sigma_intercept,
                dims="session",
            )

            session_slope_scaled = pm.Normal(
                "session_slope_scaled",
                mu=0.0,
                sigma=1.0,
                shape=num_sessions,
                dims="session",
            )
            session_slope = pm.Deterministic(
                "session_slope",
                mu_slope + session_slope_scaled * sigma_slope,
                dims="session",
            )

        b_os_rf = pm.Normal("b_os_rf", mu=0.0, sigma=1.0)
        b_log_fr = pm.Normal("b_log_fr", mu=0.0, sigma=1.0)


        yest = (
            # linear model for median, since we fit to log-data, we have to log-trafo
            # out linear slopes and intercepts
            pm.math.log(
                # session-level intercept. one for each sessions
                session_intercept[session_idx]
                # session-level slope x neuron hierarchy score
                + session_slope[session_idx]
                * df["z_hierarchy_score"].values
            )
            # per-unit terms
            + b_os_rf * df["on_screen_rf"].values 
            + b_log_fr * df["z_log_firing_rate"].values
        )

        # define normal likelihood with halfnormal error
        epsilon = pm.HalfCauchy("epsilon", 10.0)

        if measure == "R_tot":
            likelihood = pm.Normal(
                "likelihood",
                mu=yest,
                sigma=epsilon,
                observed=df[f"z_log_{measure}"],
                dims="datapoint",
            )
        else:
            alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
            likelihood = pm.SkewNormal(
                "likelihood",
                mu=yest,
                sigma=epsilon,
                alpha=alpha,
                observed=df[f"z_log_{measure}"],
                dims="datapoint",
            )


class LinearModel(ModelBase):
    def __init__(self, df, measure, noncentered=True, name=""):

        # this copies df, prepares and assigns self.df
        df = self.prepare_data(df)

        # to model session-level details, we need a lookup row -> session_idx
        session_idx, sessions = pd.factorize(df["session"])
        df["session_idx"] = session_idx
        num_sessions = len(sessions)

        # it helps readability of arviz to define dimensionality via coords:
        # https://www.pymc.io/projects/docs/en/v5.10.2/learn/core_notebooks/dimensionality.html#dims
        # this enables e.g. `idata.sel(session=0)`
        coords = {
            # we could use the real sessions of the data, but this makes it harder to index.
            "session": np.unique(session_idx),
            "datapoint": np.arange(len(df)),
        }

        super().__init__(name=name, model=None, coords=coords)

        # hyperpriors
        mu_intercept = pm.Normal("mu_intercept", mu=0.0, sigma=1.0)
        sigma_intercept = pm.HalfCauchy("sigma_intercept", beta=1.0)

        mu_slope = pm.Normal("mu_slope", mu=0.0, sigma=1.0)
        sigma_slope = pm.HalfCauchy("sigma_slope", beta=1.0)

        if noncentered is False:
            session_intercept = pm.Normal(
                "session_intercept",
                mu=mu_intercept,
                sigma=sigma_intercept,
                shape=num_sessions,
                dims="session",
            )

            session_slope = pm.Normal(
                "session_slope",
                mu=mu_slope,
                sigma=sigma_slope,
                shape=num_sessions,
                dims="session",
            )
        else:
            # avoid funnel of hell
            # https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
            session_intercept_scaled = pm.Normal(
                "session_intercept_scaled",
                mu=0.0,
                sigma=1.0,
                shape=num_sessions,
                dims="session",
            )
            session_intercept = pm.Deterministic(
                "session_intercept",
                mu_intercept + session_intercept_scaled * sigma_intercept,
                dims="session",
            )

            session_slope_scaled = pm.Normal(
                "session_slope_scaled",
                mu=0.0,
                sigma=1.0,
                shape=num_sessions,
                dims="session",
            )
            session_slope = pm.Deterministic(
                "session_slope",
                mu_slope + session_slope_scaled * sigma_slope,
                dims="session",
            )

        b_os_rf = pm.Normal("b_os_rf", mu=0.0, sigma=1.0)
        b_log_fr = pm.Normal("b_log_fr", mu=0.0, sigma=1.0)

        # linear model
        yest = (
            # session-level intercept. one for each sessions
            session_intercept[session_idx]
            # session-level slope x neuron hierarchy score
            + session_slope[session_idx]
            * df["z_hierarchy_score"].values
            # per-unit terms
            + b_os_rf * df["on_screen_rf"].values
            + b_log_fr * df["z_log_firing_rate"].values
        )

        # define normal likelihood with halfnormal error
        epsilon = pm.HalfCauchy("epsilon", 10.0)

        if measure == "R_tot":
            likelihood = pm.Normal(
                "likelihood",
                mu=yest,
                sigma=epsilon,
                observed=df[f"z_log_{measure}"],
                dims="datapoint",
            )
        else:
            alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
            likelihood = pm.SkewNormal(
                "likelihood",
                mu=yest,
                sigma=epsilon,
                alpha=alpha,
                observed=df[f"z_log_{measure}"],
                dims="datapoint",
            )


class StructureGroupModel(ModelBase):
    def __init__(self, df, measure, noncentered=True, name=""):

        df = df.copy()

        # for the structure group model, we need to select data from higher cortical and thalamus, respectively.
        # add a column each that is (1) if the neuron is in this group else (0)
        df["is_higher_cortical"] = (
            df["structure_name"].isin(["LM", "RL", "AL", "PM", "AM"]).astype("int")
        )
        df["is_thalamus"] = df["structure_name"].isin(["LGN", "LP"]).astype("int")

        log.info(df["is_higher_cortical"].value_counts())
        log.info(df["is_thalamus"].value_counts())

        df = self.prepare_data(df)

        # to model session-level details, we need a lookup row -> session_idx
        session_idx, sessions = pd.factorize(df["session"])
        df["session_idx"] = session_idx
        num_sessions = len(sessions)

        # it helps readability of arviz to define dimensionality via coords:
        # https://www.pymc.io/projects/docs/en/v5.10.2/learn/core_notebooks/dimensionality.html#dims
        # this enables e.g. `idata.sel(session=0)`
        coords = {
            # we could use the real sessions of the data, but this makes it harder to index.
            "session": np.unique(session_idx),
            "datapoint": np.arange(len(df)),
        }

        super().__init__(name=name, model=None, coords=coords)

        # hyperpriors
        mu_intercept = pm.Normal("mu_intercept", mu=0.0, sigma=1.0)
        sigma_intercept = pm.HalfCauchy("sigma_intercept", beta=0.1)

        # higher cortical
        mu_hc_offset = pm.Normal("mu_hc_offset", mu=0.0, sigma=1.0)
        sigma_hc_offset = pm.HalfCauchy("sigma_hc_offset", beta=1.0)

        # thalamus
        mu_th_offset = pm.Normal("mu_th_offset", mu=0.0, sigma=1.0)
        sigma_th_offset = pm.HalfCauchy("sigma_th_offset", beta=1.0)

        if noncentered is False:
            session_intercept = pm.Normal(
                "session_intercept",
                mu=mu_intercept,
                sigma=sigma_intercept,
                shape=num_sessions,
                dims="session",
            )
            session_hc_offset = pm.Normal(
                "session_hc_offset",
                mu=mu_hc_offset,
                sigma=sigma_hc_offset,
                shape=num_sessions,
                dims="session",
            )
            session_th_offset = pm.Normal(
                "session_th_offset",
                mu=mu_th_offset,
                sigma=sigma_th_offset,
                shape=num_sessions,
                dims="session",
            )
        else:
            session_intercept_scaled = pm.Normal(
                "session_intercept_scaled",
                mu=0.0,
                sigma=1.0,
                shape=num_sessions,
                dims="session",
            )
            session_intercept = pm.Deterministic(
                "session_intercept",
                mu_intercept + session_intercept_scaled * sigma_intercept,
                dims="session",
            )

            session_hc_offset_scaled = pm.Normal(
                "session_hc_offset_scaled",
                mu=0.0,
                sigma=1.0,
                shape=num_sessions,
                dims="session",
            )
            session_hc_offset = pm.Deterministic(
                "session_hc_offset",
                mu_hc_offset + session_hc_offset_scaled * sigma_hc_offset,
                dims="session",
            )

            session_th_offset_scaled = pm.Normal(
                "session_th_offset_scaled",
                mu=0.0,
                sigma=1.0,
                shape=num_sessions,
                dims="session",
            )
            session_th_offset = pm.Deterministic(
                "session_th_offset",
                mu_th_offset + session_th_offset_scaled * sigma_th_offset,
                dims="session",
            )

        # global_intercept = pm.Normal("global_intercept", mu=0.0, sigma=1.0)
        b_os_rf = pm.Normal("b_os_rf", mu=0.0, sigma=1.0)
        b_log_fr = pm.Normal("b_log_fr", mu=0.0, sigma=1.0)

        # # structure group model
        # yest = (
        #     # global intercept
        #     global_intercept
        #     # session-level intercept. one for each sessions
        #     + session_intercept[session_idx]
        #     #
        #     + session_hc_offset[session_idx] * df["is_higher_cortical"].values
        #     #
        #     + session_th_offset[session_idx] * df["is_thalamus"].values
        #     # per-unit terms
        #     + b_os_rf * df["on_screen_rf"].values
        #     + b_log_fr * df["z_log_firing_rate"].values
        # )

        yest = (
            pm.math.log(
                # session-level intercept (for V1). one for each sessions
                session_intercept[session_idx]
                # offset of higher cortical areas relative to V1
                + session_hc_offset[session_idx] * df["is_higher_cortical"].values
                # offset of thalamic areas relative to V1
                + session_th_offset[session_idx] * df["is_thalamus"].values)
            # per-unit terms
            + b_os_rf * df["on_screen_rf"].values
            + b_log_fr * df["z_log_firing_rate"].values
        )

        # define normal likelihood with halfnormal error
        epsilon = pm.HalfCauchy("epsilon", 10.0)

        if measure == "R_tot":
            likelihood = pm.Normal(
                "likelihood",
                mu=yest,
                sigma=epsilon,
                observed=df[f"z_log_{measure}"],
                dims="datapoint",
            )
        else:
            alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
            likelihood = pm.SkewNormal(
                "likelihood",
                mu=yest,
                sigma=epsilon,
                alpha=alpha,
                observed=df[f"z_log_{measure}"],
                dims="datapoint",
            )
