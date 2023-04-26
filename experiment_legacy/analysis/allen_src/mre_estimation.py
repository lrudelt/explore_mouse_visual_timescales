import numpy as np
import mrestimator as mre


def f_two_timescales(k, tau1, A1, tau2, A2):
    return np.abs(A1)*np.exp(-k/tau1) + np.abs(A2)*np.exp(-k/tau2)

# tau1, A1, tau2, A2
fitpars_two_timescales = np.array([
                    (0.1, 0.01, 10, 0.01),
                    (0.1, 0.1, 10, 0.01),
                    (0.5, 0.01, 10, 0.001),
                    (0.5, 0.1, 10, 0.01),
                    (0.1, 0.01, 10, 0),
                    (0.1, 0.1, 10, 0),
                    (0.5, 0.01, 10, 0),
                    (0.5, 0.1, 10, 0)])

# tau, A, O
fitpars = np.array([(0.1, 0.01, 0),
                    (0.1, 0.1, 0),
                    (1, 0.01, 0),
                    (1, 0.1, 0)])

def single_timescale_fit(rk, fitpars = fitpars):
    fit = mre.fit(rk, fitpars=fitpars)
    return fit 

def two_timescales_fit(rk, fitpars = fitpars_two_timescales):
    fit = mre.fit(rk, fitpars=fitpars, fitfunc = f_two_timescales)
    tau_1 = fit.popt[0]
    A_1 = np.abs(fit.popt[1])
    tau_2 = fit.popt[2]
    A_2 = np.abs(fit.popt[3])
    # Choose the timescale with higher coefficient A
    tau_selected = (tau_1 ,tau_2)[np.argmax((A_1,A_2))]
    tau_rejected = (tau_1 ,tau_2)[np.argmin((A_1,A_2))]
    A_selected = np.amax((A_1,A_2))
    A_rejected = np.amin((A_1,A_2))
    return fit, tau_selected, A_selected, tau_rejected, A_rejected


def get_BIC(fit):
    K = len(fit.popt)
    N = len(fit.steps)
    var = np.sum(fit.ssres) / N
    BIC = np.log(N) * K + N * np.log(var)
    return BIC

def get_AIC(fit):
    K = len(fit.popt)
    N = len(fit.steps)
    var = np.sum(fit.ssres) / N
    AIC = 2 * K + N * np.log(var)
    return AIC

def test_BIC(fit, rk): #Sort out units that have very shallow autocorrelation
    BIC_fit = get_BIC(fit)
    N = len(fit.steps)
    var_constant = np.var(rk.coefficients) # variance around a constant mean/offset of the AC
    BIC_constant = np.log(N) + N * np.log(var_constant)
    if BIC_fit < BIC_constant:
        BIC_passed = True
    else:
        BIC_passed = False
    return BIC_fit, BIC_passed

def test_AIC(fit, rk):
    AIC_fit = get_AIC(fit)
    N = len(fit.steps)
    var_constant = np.var(rk.coefficients) # variance around a constant mean/offset of the AC
    AIC_constant = 2 + N * np.log(var_constant)
    if AIC_fit < AIC_constant:
        AIC_passed = True
    else:
        AIC_passed = False
    return AIC_fit, AIC_passed