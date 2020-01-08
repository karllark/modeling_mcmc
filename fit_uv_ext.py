import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.fitting import (Fitter,
                                      _validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params,
                                      _convert_input)
from astropy.modeling.optimizers import Optimization
# from astropy.modeling.statistic import leastsquare

from dust_extinction.averages import GCC09_MWAvg
from dust_extinction.shapes import FM90


def leastsquare(measured_vals, updated_model, weights, x, y=None):
    """
    Least square statistic with optional weights.

    Parameters
    ----------
    measured_vals : `~numpy.ndarray`
        Measured data values.
    updated_model : `~astropy.modeling.Model`
        Model with parameters set by the current iteration of the optimizer.
    weights : `~numpy.ndarray`
        Array of weights to apply to each residual.
    x : `~numpy.ndarray`
        Independent variable "x" to evaluate the model on.
    y : `~numpy.ndarray`, optional
        Independent variable "y" to evaluate the model on, for 2D models.

    Returns
    -------
    res : float
        The sum of least squares.
    """

    if y is None:
        model_vals = updated_model(x)
    else:
        model_vals = updated_model(x, y)
    if weights is None:
        return np.sum((model_vals - measured_vals) ** 2)
    else:
        return np.sum((weights * (model_vals - measured_vals)) ** 2)


class ScipyMinimize(Optimization):
    """
    Interface to sciypy.minimize
    """
    # supported_constraints = ['bounds', 'eqcons', 'ineqcons', 'fixed', 'tied']

    def __init__(self):
        from scipy.optimize import minimize
        super().__init__(minimize)
        self.fit_info = {
        }

    def __call__(self, objfunc, initval, fargs, **kwargs):
        """
        Run the solver.

        Parameters
        ----------
        objfunc : callable
            objection function
        initval : iterable
            initial guess for the parameter values
        fargs : tuple
            other arguments to be passed to the statistic function
        kwargs : dict
            other keyword arguments to be passed to the solver

        """
        optresult = self.opt_method(objfunc, initval, args=fargs)
        fitparams = optresult['x']
        # print(optresult)

        return fitparams, self.fit_info


class SPyMinimizeFitter(Fitter):
    """
    Use scipy.minimize and least squares statistic
    """

    def __init__(self):
        super().__init__(optimizer=ScipyMinimize, statistic=leastsquare)
        self.fit_info = {}

    def __call__(self, model, x, y, weights=None, **kwargs):
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y
        x : array
            input coordinates
        y : array
            input coordinates
        weights : array, optional
            Weights for fitting.
            For data with Gaussian uncertainties, the weights should be
            1/sigma.
        kwargs : dict
            optional keyword arguments to be passed to the optimizer or the statistic

        Returns
        -------
        model_copy : `~astropy.modeling.FittableModel`
            a copy of the input model with parameters set by the fitter
        """
        model_copy = _validate_model(model, self._opt_method.supported_constraints)
        farg = _convert_input(x, y)
        farg = (model_copy, weights, ) + farg
        p0, _ = _model_to_fit_params(model_copy)
        fitparams, self.fit_info = self._opt_method(
            self.objective_function, p0, farg, **kwargs)
        _fitter_to_model_params(model_copy, fitparams)

        return model_copy


class EmceeOpt(Optimization):
    """
    Interface to emcee sampler.
    """
    # supported_constraints = ['bounds', 'eqcons', 'ineqcons', 'fixed', 'tied']

    def __init__(self):
        import emcee
        super().__init__(emcee)
        self.fit_info = {
            'perparams': None,
            'samples': None,
            'sampler': None
        }

    @staticmethod
    def _get_best_fit_params(sampler):
        """
        Determine the best fit parameters given an emcee sampler object
        """
        # very likely a faster way
        max_lnp = -1e6
        nwalkers = len(sampler.lnprobability)
        for k in range(nwalkers):
            tmax_lnp = np.max(sampler.lnprobability[k])
            if tmax_lnp > max_lnp:
                max_lnp = tmax_lnp
                indxs, = np.where(sampler.lnprobability[k] == tmax_lnp)
                fit_params_best = sampler.chain[k, indxs[0], :]

        return fit_params_best

    @staticmethod
    def _get_percentile_params(sampler):
        """
        Determine the 50p plus/minus 34p vlaues
        """
        nwalkers = len(sampler.lnprobability)
        # discard the 1st 10% (burn in)
        flat_samples = sampler.get_chain(discard=int(0.1 * nwalkers), flat=True)
        nwalkers, ndim = flat_samples.shape

        per_params = []
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            per_params.append(mcmc)

        return list(per_params)

    def __call__(self, objfunc, initval, fargs, nsteps, **kwargs):
        """
        Run the sampler.

        Parameters
        ----------
        objfunc : callable
            objection function
        initval : iterable
            initial guess for the parameter values
        fargs : tuple
            other arguments to be passed to the statistic function
        kwargs : dict
            other keyword arguments to be passed to the solver
        """
        # optresult = self.opt_method(objfunc, initval, args=fargs)
        # fitparams = optresult['x']

        ndim = len(initval)
        nwalkers = 2 * ndim
        pos = initval + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = self.opt_method.EnsembleSampler(nwalkers, ndim, objfunc,
                                                  args=fargs)
        sampler.run_mcmc(pos, nsteps, progress=True)
        samples = sampler.get_chain()

        fitparams = self._get_best_fit_params(sampler)
        self.fit_info['perparams'] = self._get_percentile_params(sampler)
        self.fit_info['sampler'] = sampler
        self.fit_info['samples'] = samples

        return fitparams, self.fit_info


class EmceeFitter(Fitter):
    """
    Use emcee and least squares statistic
    """

    def __init__(self, nsteps=100):
        super().__init__(optimizer=EmceeOpt, statistic=leastsquare)
        self.nsteps = nsteps
        self.fit_info = {}

    def log_probability(self, fps, *args):
        """
        Get the log probability

        Parameters
        ----------
        fps : list
            parameters returned by the fitter
        args : list
            [model, [other_args], [input coordinates]]
            other_args may include weights or any other quantities specific for
            a statistic

        Notes
        -----
        The list of arguments (args) is set in the `__call__` method.
        Fitters may overwrite this method, e.g. when statistic functions
        require other arguments.
        """
        # get standard leastsquare
        res = self.objective_function(fps, *args)

        # convert to a log probability - assumes chisqr/Gaussian unc model
        return -0.5 * res

    def __call__(self, model, x, y, weights=None, **kwargs):
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y
        x : array
            input coordinates
        y : array
            input coordinates
        weights : array, optional
            Weights for fitting.
            For data with Gaussian uncertainties, the weights should be
            1/sigma.
        kwargs : dict
            optional keyword arguments to be passed to the optimizer or the statistic

        Returns
        -------
        model_copy : `~astropy.modeling.FittableModel`
            a copy of the input model with parameters set by the fitter
        """

        model_copy = _validate_model(model, self._opt_method.supported_constraints)
        farg = _convert_input(x, y)
        farg = (model_copy, weights, ) + farg
        p0, _ = _model_to_fit_params(model_copy)

        fitparams, self.fit_info = self._opt_method(
            self.log_probability, p0, farg, self.nsteps, **kwargs)

        _fitter_to_model_params(model_copy, fitparams)

        return model_copy


if __name__ == '__main__':

    # get the data to fit
    mwext = GCC09_MWAvg()
    x = mwext.obsdata_x_iue
    y = mwext.obsdata_axav_iue
    y_unc = mwext.obsdata_axav_unc_iue * 10.
    gindxs = x > 3.125

    # initialize the model
    fm90_init = FM90()

    # pick the fitter
    fit = LevMarLSQFitter()
    fit2 = SPyMinimizeFitter()
    nsteps = 1000
    fit3 = EmceeFitter(nsteps=nsteps)

    # fit the data to the FM90 model using the fitter
    #   use the initialized model as the starting point
    fm90_fit = fit(fm90_init, x[gindxs], y[gindxs], weights=1.0 / y_unc[gindxs])
    fm90_fit2 = fit2(fm90_init, x[gindxs], y[gindxs], weights=1.0 / y_unc[gindxs])
    fm90_fit3 = fit3(fm90_fit2, x[gindxs], y[gindxs], weights=1.0 / y_unc[gindxs])
    # print(fit3.fit_info['perparams'])

    # print(fit3.fit_info['sampler'].get_autocorr_time())

    # plot the observed data, initial guess, and final fit
    fig, ax = plt.subplots()

    # ax.errorbar(x, y, yerr=y_unc[gindxs], fmt='ko', label='Observed Curve')
    # ax.plot(x[gindxs], fm90_init(x[gindxs]), label='Initial guess')
    ax.plot(x, y, label='Observed Curve')
    ax.plot(x[gindxs], fm90_fit3(x[gindxs]), label='emcee')
    ax.plot(x[gindxs], fm90_fit2(x[gindxs]), label='scipy.minimize')
    ax.plot(x[gindxs], fm90_fit(x[gindxs]), label='LevMarLSQ')

    # plot samples from the mcmc chaing
    flat_samples = fit3.fit_info['sampler'].get_chain(discard=int(0.1 * nsteps), flat=True)
    inds = np.random.randint(len(flat_samples), size=100)
    model_copy = fm90_fit3.copy()
    for ind in inds:
        sample = flat_samples[ind]
        _fitter_to_model_params(model_copy, sample)
        plt.plot(x[gindxs], model_copy(x[gindxs]), "C1", alpha=0.05)

    ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
    ax.set_ylabel('$A(x)/A(V)$')

    ax.set_title('FM90 Fit to G09_MWAvg curve')

    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
