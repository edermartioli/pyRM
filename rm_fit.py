# -*- coding: iso-8859-1 -*-
"""
    Created on January 31 2020
    
    Description: Fit Rossiter-McLaughlin to RV data
    
    @author: Eder Martioli <martioli@iap.fr>, Shweta Dalal <dalal@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python ./rm_fit.py --pattern=data/HD189733*.rdb --exoplanet_priors=HD189733.pars --calib_priors= -v

    python ./rm_fit.py --pattern=data/HD209458_smm.rdb --exoplanet_priors=priors/HD209458.pars -v
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """


from optparse import OptionParser
import os,sys

import numpy as np
import glob
import rm_lib
import priorslib
from rm_lib import lnprob
import matplotlib.pyplot as plt
from astropy.io import ascii
import emcee

parser = OptionParser()
parser.add_option("-p", "--pattern", dest="pattern", help="RV data pattern",type='string',default="*.rdb")
parser.add_option("-e", "--exoplanet_priors", dest="exoplanet_priors", help="File containing exoplanet priors",type='string',default="")
parser.add_option("-c", "--calib_priors", dest="calib_priors", help="File containing calibration priors",type='string',default="")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with rm_fit.py -h ")
    sys.exit(1)

if options.verbose:
    print('RV data pattern: ', options.pattern)
    print('Exoplanet priors file: ', options.exoplanet_priors)
    print('Calibration priors file: ', options.calib_priors)

planet_posterior = (options.exoplanet_priors).replace(".pars", "_posterior.pars")
if options.verbose:
    print("Output PLANET posterior: ", planet_posterior)

if options.calib_priors != "":
    calib_posterior = (options.calib_priors).replace(".pars", "_posterior.pars")
else :
    calib_posterior = "calibration_posterior.pars"

if options.verbose:
    print("Output CALIBRATION posterior: ", calib_posterior)

# make list of tfits data files
if options.verbose:
    print("Creating list of RV time series files...")
inputdata = sorted(glob.glob(options.pattern))

# Load data
bjd, rvs, rverrs  = [], [], []
for i in range(len(inputdata)) :
    data = ascii.read(inputdata[i],data_start=2)
    bjd.append(np.array(data['rjd'] + 2400000.))
    rvs.append(np.array(data["vrad"]))
    rverrs.append(np.array(data["svrad"]))

# Load priors:
# Load exoplanet parameters priors
planet_priors = priorslib.read_priors(options.exoplanet_priors)
planet_params = priorslib.read_exoplanet_params(planet_priors)

# print out planet priors
if options.verbose:
    print("----------------")
    print("Input PLANET parameters:")
    for key in planet_params.keys() :
        print(key, "=", planet_params[key])
    print("----------------")

# if priors file is provided then load calibration parameters priors
if options.calib_priors != "" :
    calib_priors = priorslib.read_priors(options.calib_priors, calibration = True)
    calib_params = priorslib.read_calib_params(calib_priors)
else :
#if no priors file is provided then make a guess
    calib_priors = priorslib.init_calib_priors(ndim=len(inputdata), order=3)
    calib_params = priorslib.read_calib_params(calib_priors)
    for i in range(len(rvs)) :
        coeff_name = 'd{0:02d}c0'.format(i)
        calib_params[coeff_name] = np.mean(rvs[i]) - np.mean(rm_lib.rv_model(planet_params, bjd[i]))
        calib_priors[coeff_name]['object'].value = np.mean(rvs[i]) - np.mean(rm_lib.rv_model(planet_params, bjd[i]))

# print out calibration priors
if options.verbose:
    print("----------------")
    print("Input CALIBRATION parameters:")
    for key in calib_params.keys() :
        print(key, "=", calib_params[key])
    print("----------------")

# Variable "theta" stores only the free parameters, and "labels" stores the corresponding parameter IDs
theta, labels, theta_priors = priorslib.get_theta_from_priors(planet_priors, calib_priors)

# Print out free parameters, i.e., theta parameters
if options.verbose:
    print("----------------")
    print("Free parameters:")
    for i in range(len(theta)) :
        
        
        print(labels[i], " -> initial guess:", theta[i])
    
    print("----------------")


"""
# Uncomment this part to plot model and data for the intial guess
for i in range(len(bjd)) :
    calib = rm_lib.calib_model(len(bjd), i, calib_params, bjd[i])
    rvcurve = rm_lib.rv_model(planet_params, bjd[i])
    plt.plot(bjd[i],rvcurve + calib, '-')
    plt.errorbar(bjd[i], rvs[i], yerr=rverrs[i], fmt='o')
plt.show()
"""

#- initialize emcee sampler
amp, ndim, nwalkers, niter, burnin = 5e-4, len(theta), 32, 2000, 500
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [theta_priors, labels, calib_params, planet_params, bjd, rvs, rverrs])
pos = [theta + amp * np.random.randn(ndim) for i in range(nwalkers)]
#--------

#- run mcmc
if options.verbose:
    print("Running MCMC ...")
sampler.run_mcmc(pos, niter)
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) # burnin : number of first samples to be discard as burn-in
#--------

# Obtain best fit calibration parameters from pdfs
calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err = rm_lib.best_fit_params(calib_params, labels, samples)

# Obtain best fit planet parameters from pdfs
planet_params, planet_theta_fit, planet_theta_labels, planet_theta_err = rm_lib.best_fit_params(planet_params, labels, samples)

# print out best fit parameters and errors
if options.verbose:
    print("----------------")
    print("PLANET Fit parameters:")
    for i in range(len(planet_theta_fit)) :
        if planet_theta_labels[i] == 'lambda' :
            lambda_deg = planet_theta_fit[i] * 180. / np.pi
            lambda_err_deg = np.array(planet_theta_err[i]) * 180. / np.pi
            print("lambda (deg) =", lambda_deg, "+", lambda_err_deg[0], "-", lambda_err_deg[1])
        print(planet_theta_labels[i], "=", planet_theta_fit[i], "+", planet_theta_err[i][0], "-", planet_theta_err[i][1])

    print("----------------")
    print("CALIBRATION Fit parameters:")
    for i in range(len(calib_theta_fit)) :
        print(calib_theta_labels[i], "=", calib_theta_fit[i], "+", calib_theta_err[i][0], "-", calib_theta_err[i][1])
    print("----------------")

#bjd_limits=[2458651.85,2458652.20]
bjd_limits=[]

# plot each dataset with the best model
for i in range(len(bjd)) :
    rm_lib.plot_individual_datasets(bjd, rvs, rverrs, i, planet_params, calib_params, samples, labels,bjd_limits=bjd_limits, detach_calib=False)

# save posterior of planet parameters into file:
priorslib.save_posterior(planet_posterior, planet_params, planet_theta_fit, planet_theta_labels, planet_theta_err)

# save posterior of calibration parameters into file:
ncoeff=calib_priors['orderOfPolynomial']['object'].value
priorslib.save_posterior(calib_posterior, calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err, calib=True, ncoeff=ncoeff)

#- make a pairs plot from MCMC output:
rm_lib.pairs_plot(samples, labels)
#--------

#- perform analysis of residuals:
rm_lib.analysis_of_residuals(bjd, rvs, rverrs, planet_params, calib_params, output="")
#--------
