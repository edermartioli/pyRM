# -*- coding: iso-8859-1 -*-
"""
    Created on January 31 2020
    
    Description: Fit Rossiter-McLaughlin to RV data
    
    @author: Eder Martioli <martioli@iap.fr>, Shweta Dalal <dalal@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python ./rm_fit.py --input=data/HD189733*.rdb --exoplanet_priors=priors/HD189733.pars --calib_order=1  --nsteps=400 --walkers=30 --burnin=150 --samples_filename=hd189733_mcmc_samples.h5 -v

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
parser.add_option("-i", "--input", dest="input", help="Input RV data files",type='string',default="*.rdb")
parser.add_option("-t", "--output", dest="output", help="output files",type='string',default="")
parser.add_option("-m", "--basename", dest="basename", help="basename for output files",type='string',default="")
parser.add_option("-e", "--exoplanet_priors", dest="exoplanet_priors", help="File containing exoplanet priors",type='string',default="")
parser.add_option("-c", "--calib_priors", dest="calib_priors", help="File containing calibration priors",type='string',default="")
parser.add_option("-o", "--calib_order", dest="calib_order", help="Order of calibration polynomial",type='string',default="1")
parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='string',default="300")
parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='string',default="32")
parser.add_option("-b", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='string',default="100")
parser.add_option("-s", "--samples_filename", dest="samples_filename", help="MCMC samples file name",type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)
parser.add_option("-k", action="store_true", dest="save_plots", help="save_plots", default=False)
parser.add_option("-r", action="store_true", dest="save_residuals", help="save_residuals", default=False)
parser.add_option("-g", "--timerange", dest="timerange", help="add time range for alldatasets plot (in days)",type='string',default="0")
parser.add_option("-f", "--format", dest="fmt", help="format of the outpur plots",type='string',default="png")
parser.add_option("-l", "--res", dest="res", help="alldataset plot resolution",type='string',default="m")

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with rm_fit.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input RV data files: ', options.input)
    print('Exoplanet priors file: ', options.exoplanet_priors)
    print('Calibration priors file: ', options.calib_priors)
    print('Order of calibration: ', options.calib_order)
    print('Number of MCMC steps: ', options.nsteps)
    print('Number of MCMC walkers: ', options.walkers)
    print('Number of MCMC burn-in samples: ', options.burnin)
    print('MCMC samples file name: ', options.samples_filename)
    print('output directory : ', options.output)

calib_order = int(options.calib_order)
od = options.output
inf = options.input
tr = float(options.timerange)
fmt = options.fmt
res = options.res

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
inputdata = sorted(glob.glob(options.input))
if options.basename != "":
    bn = options.basename
else:
    bn = ""
    for char in inputdata[0].rsplit('/')[-1]:
        if char.isalnum():
            bn += char
        if char in ["_","-"]:
            bn += char
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
        if ("_err" not in key) and ("_pdf" not in key) :
            pdf_key = "{0}_pdf".format(key)
            if planet_params[pdf_key] == "FIXED" :
                print("{0} = {1} ({2})".format(key, planet_params[key], planet_params[pdf_key]))
            elif planet_params[pdf_key] == "Uniform" or planet_params[pdf_key] == "Jeffreys":
                error_key = "{0}_err".format(key)
                min = planet_params[error_key][0]
                max = planet_params[error_key][1]
                print("{0} <= {1} <= {2} ({3})".format(min, key, max, planet_params[pdf_key]))
            elif planet_params[pdf_key] == "Normal" :
                error_key = "{0}_err".format(key)
                error = planet_params[error_key][1]
                print("{0} = {1} +- {2} ({3})".format(key, planet_params[key], error, planet_params[pdf_key]))
    print("----------------")

# if priors file is provided then load calibration parameters priors
if options.calib_priors != "" :
    calib_priors = priorslib.read_priors(options.calib_priors, calibration = True)
    calib_params = priorslib.read_calib_params(calib_priors)
else :
#if no priors file is provided then make a guess
    tc = planet_priors['tau']['object'].value
    calib_priors = priorslib.init_calib_priors(tc, ndim=len(inputdata), order=calib_order)
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
exit()
"""

#- initialize emcee sampler
amp, ndim, nwalkers, niter, burnin = 5e-4, len(theta), int(options.walkers), int(options.nsteps), int(options.burnin)

# Set up the backend
if options.samples_filename != "":
    backend = emcee.backends.HDFBackend(od+options.samples_filename)
    backend.reset(nwalkers, ndim)
else :
    backend = None

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [theta_priors, labels, calib_params, planet_params, bjd, rvs, rverrs], backend=backend)
pos = [theta + amp * np.random.randn(ndim) for i in range(nwalkers)]
#--------

#- run mcmc
if options.verbose:
    print("Running MCMC ...")
sampler.run_mcmc(pos, niter, progress=True)
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) # burnin : number of first samples to be discard as burn-in
#--------

# Obtain best fit calibration parameters from pdfs
calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err = rm_lib.best_fit_params(calib_params, labels, samples)

# Obtain best fit planet parameters from pdfs
planet_params, planet_theta_fit, planet_theta_labels, planet_theta_err = rm_lib.best_fit_params(planet_params, labels, samples)

p = options.plot
k = options.save_plots
r = options.save_residuals

#saves the output
if k:
    f = open(od+bn+"Parameters output.txt",'w')
    f.truncate(0)

# print out best fit parameters and errors
if options.verbose:
    print("----------------")
    print("PLANET Fit parameters:")
    f.write("PLANET Fit parameters:\n")
    for i in range(len(planet_theta_fit)) :
        print(planet_theta_labels[i], "=", planet_theta_fit[i], "+", planet_theta_err[i][0], "-", planet_theta_err[i][1])
        f.write(str(planet_theta_labels[i])+" = " +str(planet_theta_fit[i])+" + "+str(planet_theta_err[i][0])+" - "+str(planet_theta_err[i][1])+"\n")

    print("----------------")
    f.write("----------------\n")
    print("CALIBRATION Fit parameters:")
    f.write("CALIBRATION Fit parameters:\n")
    for i in range(len(calib_theta_fit)) :
        print(calib_theta_labels[i], "=", calib_theta_fit[i], "+", calib_theta_err[i][0], "-", calib_theta_err[i][1])
        f.write(str(calib_theta_labels[i])+" = "+str(calib_theta_fit[i])+" + "+str(calib_theta_err[i][0])+" - "+str(calib_theta_err[i][1])+'\n')
    print("----------------")
    f.write("----------------\n")

bjd_limits=[]


if p or k :
    print("Plotting :")
    #plot all data sets
    rm_lib.plot_all_datasets(bjd, rvs, rverrs, planet_params, calib_params, samples, labels, theta_priors, tr, tr, fmt, res, bn, p, k, od)

    # plot each dataset with the best model
    for i in range(len(bjd)) :
        rm_lib.plot_individual_datasets(bjd, rvs, rverrs, i, planet_params, calib_params, samples, labels, theta_priors, fmt,inf, p, k, od, bjd_limits=bjd_limits, detach_calib=False)

# save posterior of planet parameters into file:
priorslib.save_posterior(planet_posterior, planet_params, planet_theta_fit, planet_theta_labels, planet_theta_err)

# save posterior of calibration parameters into file:
ncoeff=calib_priors['orderOfPolynomial']['object'].value
priorslib.save_posterior(calib_posterior, calib_params, calib_theta_fit, calib_theta_labels, calib_theta_err, calib=True, ncoeff=ncoeff)

if p or k :
    #- make a pairs plot from MCMC output:
    rm_lib.pairs_plot(samples, labels, calib_params, planet_params, fmt, bn, p, k, od)
    #--------

if p or k or r:
    #- perform analysis of residuals:
    print("----------------")
    print("Running Analysis of residuals:")
    f.write("Analysis of residuals :\n----------------\n")
    f.close()
    rm_lib.analysis_of_residuals(bjd, rvs, rverrs, planet_params, calib_params, theta_priors, bn, fmt, inf, p, k, r, od, output="")
    #--------
