import matplotlib.pyplot as plt
import numpy as np
import argparse

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.fitting import (
    Fitter,
    _validate_model,
    _fitter_to_model_params,
    _model_to_fit_params,
    _convert_input,
)
from astropy.modeling.optimizers import Optimization
import astropy.units as u
from astropy.modeling.statistic import leastsquare
from astropy import uncertainty as astrounc

import corner

from dust_extinction.averages import GCC09_MWAvg
from dust_extinction.shapes import FM90

from measure_extinction.extdata import ExtData


class ScipyMinimize(Optimization):
    """
    Interface to sciypy.minimize
    """

    # supported_constraints = ['bounds', 'eqcons', 'ineqcons', 'fixed', 'tied']

    def __init__(self):
        from scipy.optimize import minimize

        super().__init__(minimize)
        self.fit_info = {}

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
        fitparams = optresult["x"]
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
        farg = (model_copy, weights) + farg
        p0, _ = _model_to_fit_params(model_copy)
        fitparams, self.fit_info = self._opt_method(
            self.objective_function, p0, farg, **kwargs
        )
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
        self.fit_info = {"perparams": None, "samples": None, "sampler": None}

    @staticmethod
    def _get_best_fit_params(sampler):
        """
        Determine the best fit parameters given an emcee sampler object
        """
        # very likely a faster way
        max_lnp = -1e6
        nwalkers, nsteps = sampler.lnprobability.shape
        for k in range(nwalkers):
            tmax_lnp = np.max(sampler.lnprobability[k])
            if tmax_lnp > max_lnp:
                max_lnp = tmax_lnp
                indxs, = np.where(sampler.lnprobability[k] == tmax_lnp)
                fit_params_best = sampler.chain[k, indxs[0], :]

        return fit_params_best

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

        sampler = self.opt_method.EnsembleSampler(nwalkers, ndim, objfunc, args=fargs)
        sampler.run_mcmc(pos, nsteps, progress=True)
        samples = sampler.get_chain()

        fitparams = self._get_best_fit_params(sampler)
        self.fit_info["sampler"] = sampler
        self.fit_info["samples"] = samples

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

    def _set_uncs_and_posterior(self, model, burn_frac=0.1):
        """
        Set the symmetric and asymmetric Gaussian uncertainties
        TBD: how to set these on the parameter objects not just the model arrays

        Parameters
        ----------
        model : astropy model
            model giving the result from the fitting

        burn_frac : float
            burn in fraction to ignore in unc calculations

        Returns
        -------
        model : astropy model
            model updated with uncertainties
        """
        sampler = self.fit_info['sampler']
        nwalkers, nsteps = sampler.lnprobability.shape
        # discard the 1st burn_frac (burn in)
        flat_samples = sampler.get_chain(discard=int(burn_frac * nsteps), flat=True)
        nflatsteps, ndim = flat_samples.shape

        model.uncs = np.zeros((ndim))
        model.uncs_plus = np.zeros((ndim))
        model.uncs_minus = np.zeros((ndim))
        for i, pname in enumerate(model.param_names):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])

            # set the uncertainty arrays - could be done via the parameter objects
            # but would need an update to the model properties to make this happen
            model.uncs[i] = 0.5 * (mcmc[2] - mcmc[0])
            model.uncs_plus[i] = mcmc[2] - mcmc[1]
            model.uncs_minus[i] = mcmc[1] - mcmc[0]

            # now set uncertainties on the parameter objects themselves
            param = getattr(model, pname)
            param.unc = model.uncs[i]
            param.unc_plus = model.uncs_plus[i]
            param.unc_minus = model.uncs_minus[i]

            # set the posterior distribution to the samples
            param.posterior = astrounc.Distribution(flat_samples[:, i])

        return model

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
        farg = (model_copy, weights) + farg
        p0, _ = _model_to_fit_params(model_copy)

        fitparams, self.fit_info = self._opt_method(
            self.log_probability, p0, farg, self.nsteps, **kwargs
        )

        # set the output model parameters to the "best fit" parameters
        _fitter_to_model_params(model_copy, fitparams)

        # get and set the symmetric and asymmetric uncertainties on each parameter
        model_copy = self._set_uncs_and_posterior(model_copy)

        return model_copy


def plot_emcee_results(sampler, fit_param_names, filebase=""):
    """
    Plot the standard triangle and diagnostic walker plots
    """

    # plot the walker chains for all parameters
    nwalkers, nsteps, ndim = sampler.chain.shape
    fig, ax = plt.subplots(ndim, sharex=True, figsize=(13, 13))
    walk_val = np.arange(nsteps)
    for i in range(ndim):
        for k in range(nwalkers):
            ax[i].plot(walk_val, sampler.chain[k, :, i], "-")
            ax[i].set_ylabel(fit_param_names[i])
    fig.savefig("%s_walker_param_values.png" % filebase)
    plt.close(fig)

    # plot the 1D and 2D likelihood functions in a traditional triangle plot
    samples = sampler.chain.reshape((-1, ndim))
    fig = corner.corner(
        samples,
        labels=fit_param_names,
        show_titles=True,
        title_fmt=".3f",
        use_math_text=True,
    )
    fig.savefig("%s_param_triangle.png" % filebase)
    plt.close(fig)


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("extfile", help="file with extinction curve")
    parser.add_argument(
        "--nsteps", type=int, default=1000, help="# of steps in MCMC chain"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get a saved extnction curve
    file = args.extfile
    # file = '/home/kgordon/Python_git/spitzer_mir_ext/fits/hd147889_hd064802_ext.fits'
    ofile = file.replace(".fits", "_fm90.fits")
    ext = ExtData(filename=file)
    ext.trans_elv_alav(av=float(ext.columns['AV'][0]))
    gindxs = ext.npts["IUE"] > 0
    x = 1.0 / ext.waves["IUE"][gindxs].to(u.micron).value
    y = ext.exts["IUE"][gindxs]
    y_unc = ext.uncs["IUE"][gindxs]
    gindxs = (x > 3.5) & (x < 8.0)

    # initialize the model
    fm90_init = FM90()

    # pick the fitter
    fit = LevMarLSQFitter()
    fit2 = SPyMinimizeFitter()
    nsteps = args.nsteps
    fit3 = EmceeFitter(nsteps=nsteps)

    # fit the data to the FM90 model using the fitter
    #   use the initialized model as the starting point
    fm90_fit = fit(fm90_init, x[gindxs], y[gindxs], weights=1.0 / y_unc[gindxs])
    fm90_fit2 = fit2(fm90_init, x[gindxs], y[gindxs], weights=1.0 / y_unc[gindxs])
    fm90_fit3 = fit3(fm90_fit, x[gindxs], y[gindxs], weights=1.0 / y_unc[gindxs])

    # checking the uncertainties
    print("Best Fit Parameters")
    print(fm90_fit3.parameters)
    print("Symmetric uncertainties")
    print(fm90_fit3.uncs)
    print("Plus uncertainties")
    print(fm90_fit3.uncs_plus)
    print("Minus uncertainties")
    print(fm90_fit3.uncs_minus)

    for i, pname in enumerate(fm90_fit3.param_names):
        # now set uncertainties on the parameter objects themselves
        param = getattr(fm90_fit3, pname)
        print("posterior: ", pname, param.posterior.pdf_mean(), param.posterior.pdf_std())
        print("parameter: ", pname, param.value, param.unc)

    # other stuff
    fm90_best_params = (fm90_fit2.param_names, fm90_fit2.parameters)

    # percentile parameters
    samples = fit3.fit_info["sampler"].chain.reshape((-1, 6))
    per_params = [
        (v[1], v[2] - v[1], v[1] - v[0])
        for v in zip(*np.percentile(samples, [16, 50, 84], axis=0))
    ]
    fm90_per_params = (fm90_fit3.param_names, per_params)
    ext.save(ofile, fm90_best_params=fm90_best_params, fm90_per_params=fm90_per_params)

    plot_emcee_results(
        fit3.fit_info["sampler"],
        fm90_fit3.param_names,
        filebase=ofile.replace(".fits", ""),
    )

    # print(fit3.fit_info['perparams'])

    # print(fit3.fit_info['sampler'].get_autocorr_time())

    # plot the observed data, initial guess, and final fit
    fig, ax = plt.subplots()

    # ax.errorbar(x, y, yerr=y_unc[gindxs], fmt='ko', label='Observed Curve')
    # ax.plot(x[gindxs], fm90_init(x[gindxs]), label='Initial guess')
    ax.plot(x, y, label="Observed Curve")
    ax.plot(x[gindxs], fm90_fit3(x[gindxs]), label="emcee")
    ax.plot(x[gindxs], fm90_fit2(x[gindxs]), label="scipy.minimize")
    ax.plot(x[gindxs], fm90_fit(x[gindxs]), label="LevMarLSQ")

    # plot samples from the mcmc chaing
    flat_samples = fit3.fit_info["sampler"].get_chain(
        discard=int(0.1 * nsteps), flat=True
    )
    inds = np.random.randint(len(flat_samples), size=100)
    model_copy = fm90_fit3.copy()
    for ind in inds:
        sample = flat_samples[ind]
        _fitter_to_model_params(model_copy, sample)
        plt.plot(x[gindxs], model_copy(x[gindxs]), "C1", alpha=0.05)

    ax.set_xlabel(r"$x$ [$\mu m^{-1}$]")
    # ax.set_ylabel('$A(x)/A(V)$')
    ax.set_ylabel(r"$A(\lambda)/A(V)$")

    ax.set_title(file)
    # ax.set_title('FM90 Fit to G09_MWAvg curve')

    ax.legend(loc="best")
    plt.tight_layout()

    # plot or save to a file
    outname = ofile.replace(".fits", "")
    if args.png:
        fig.savefig(outname + ".png")
    elif args.pdf:
        fig.savefig(outname + ".pdf")
    else:
        plt.show()
