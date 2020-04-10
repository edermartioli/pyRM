# -*- coding: iso-8859-1 -*-
"""
    Created on January 31 2020
    
    Description: Rossiter-McLaughlin library
    
    @author: Eder Martioli <martioli@iap.fr>, Shweta Dalal <dalal@iap.fr>
    
    Institut d'Astrophysique de Paris, France.

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """


import numpy as np
import corner
import matplotlib.pyplot as plt
from scipy import stats


def vradfction(t,k,tp,omega,T0,V0,e):

    phase0=((t-T0)%tp)/tp
    phase=np.where(np.less(phase0,0),phase0+1,phase0)

    anomoy=2.*np.pi*phase
    anoexc=anomoy*1.

    anoexc1=anoexc+(anomoy+e*np.sin(anoexc)-anoexc)/(1.-e*np.cos(anoexc))
    while np.max(np.abs(anoexc1-anoexc)) > 1.e-8:
        anoexc=anoexc1*1.
        anoexc1=anoexc+(anomoy+e*np.sin(anoexc)-anoexc)/(1.-e*np.cos(anoexc))

    anovraie=2.*np.arctan(np.sqrt((1.+e)/(1.-e))*np.tan(anoexc/2.))

    vrad=V0+k*(np.cos(anovraie+omega)+e*np.cos(omega))
    
    return [vrad,phase,anovraie]

           
def gfunction(x,etap,gamma):
    fg=(1.-x**2)*np.arcsin(np.sqrt((gamma**2-(x-1.-etap)**2)/(1.-x**2)))+np.sqrt((gamma**2-(x-1.-etap)**2)*(1.-x**2-gamma**2+(x-1.-etap)**2))
    return fg


# ### derivee vitesse keplerienne car passage transit lorsque dvrad <0
#

def RManomaly(dvrad,lbda,Vs,aratio,i,Rratio,omega,eps,e,anovraie):

    
    rp=(aratio)*(1.-e**2)/((1.+e*np.cos(anovraie)))
    xp=rp*(-np.cos(lbda)*np.sin(anovraie+omega)-np.sin(lbda)*np.cos(i)*np.cos(anovraie+omega))
    zp=rp*(np.sin(lbda)*np.sin(anovraie+omega)-np.cos(lbda)*np.cos(i)*np.cos(anovraie+omega))
    R=np.sqrt(xp**2+zp**2)


    #### RV anomaly computation

    ind1=1.-Rratio
    ind2=1.+Rratio
    g=Rratio
    g2=(Rratio)**2
    v=np.zeros(len(R),'d')


    for j in range(len(R)):
        # mu = np.sqrt(1-R[j]**2)
        # print ("MU",mu)
    ## Ingress Phase and Egress phase:
        if (R[j]>ind1) and (R[j]<ind2) and dvrad[j]<0.:
            n_p=R[j]-1.
            x0=1.-(g2-n_p**2)/(2.*(1.+n_p))
            z0=np.sqrt(1.-x0**2)
            zeta=1.+n_p-x0
            xc=x0+(zeta-g)/2.
            w2=np.sqrt(1.-(1.-g)**2)
            # print ("W2,1=",w2)
            w3=0.
            w4=(np.pi/2.)*g*(g-zeta)*xc*w2*gfunction(xc,n_p,g)/gfunction(1-g,-g,g)
            v[j]=-1.*(Vs)*xp[j]*((1.-eps)*(-1.*z0*zeta+g2*np.arccos(zeta/g))+(eps*w4/(1.+n_p)))/(np.pi*(1.-(1./3.)*eps)-(1.-eps)*(np.arcsin(z0)-(1.+n_p)*z0+g2*np.arccos(zeta/g))-eps*w3)

    ## Complete transit phase:
        if (R[j]<ind1) and dvrad[j]<0.:
    #while (R[j]<ind1):
            n_p=R[j]-1.

    ## Integrals
            rho=n_p+1.
            w2=np.sqrt(1.-rho**2)
            # print("W2,2=",w2)
            w1=0.

            v[j]=-1*(Vs)*xp[j]*g2*(1.-eps*(1.-w2))/(1.-g2-eps*(1./3.-g2*(1.-w1)))

    ## Outside transit phase:
        if R[j]>ind2:
    #while (R[j]>ind2):
            v[j]=0.
    return v


def calib_model(n, i, params, bjd) :

    ncoefs = int(len(params) / n)
    
    coefs = []
    
    for c in range(int(ncoefs)):
        coeff_id = 'd{0:02d}c{1:1d}'.format(i,c)
        coefs.append(params[coeff_id])

    p = np.poly1d(np.flip(coefs))
    out_model = p(bjd)
    return out_model

def rv_model(planet_params, bjd) :
    
    per = planet_params['per']
    tau = planet_params['tau']
    k = planet_params['k']
    omega = planet_params['omega']
    ecc = planet_params['ecc']
    rv0 = planet_params['rv0']
    lambdap = planet_params['lambda']
    vsini = planet_params['vsini']
    a_R = planet_params['a_R']
    inc = planet_params['inc']
    r_R = planet_params['r_R']
    omega_rm = planet_params['omega_rm']
    ldc = planet_params['ldc']
         
    model = []

    keplerian = vradfction(bjd, k, per, omega, tau, rv0, ecc)
    
    vrad = keplerian[0]

    anovraie = keplerian[2]

    dvrad = np.concatenate((np.array([vrad[1]-vrad[0]]),vrad[1:]-vrad[:-1]))

    rm_effect = RManomaly(dvrad, lambdap, vsini, a_R, inc, r_R, omega_rm, ldc, ecc, anovraie)

    return (vrad + rm_effect)


#posterior probability
def lnprob(theta, theta_priors, labels, calib_params, planet_params, bjd, rvs, rverrs):

    #lp = lnprior(theta)
    lp = lnprior(theta_priors, theta, labels)
    if not np.isfinite(lp):
        return -np.inf
    
    prob = lp + lnlike(theta, labels, calib_params, planet_params, bjd, rvs, rverrs)

    if np.isnan(prob) :
        return -np.inf
    else :
        return prob


def updateParams(params, theta, labels) :
    for key in params.keys() :
        for j in range(len(theta)) :
            if key == labels[j]:
                params[key] = theta[j]
                break
    return params

#likelihood function
def lnlike(theta, labels, calib_params, planet_params, bjd, rvs, rverrs):
    
    planet_params = updateParams(planet_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)

    sum_of_residuals = 0
    
    for i in range(len(bjd)) :
        calib = calib_model(len(bjd), i, calib_params, bjd[i])
        rvcurve = calib + rv_model(planet_params, bjd[i])
        residuals = rvs[i] - rvcurve
        sum_of_residuals += np.sum((residuals/rverrs[i])**2 + np.log(2.0 * np.pi * (rverrs[i] * rverrs[i])))

    ln_likelihood = -0.5 * (sum_of_residuals)
    
    return ln_likelihood

'''
def lnprior(theta):
    return 0. #assumes all priors have uniform probability
'''

# prior probability from definitions in priorslib
def lnprior(theta_priors, theta, labels):

    total_prior = 0.0

    for i in range(len(theta)) :
        #theta_priors[labels[i]]['object'].set_value(theta[i])
        if theta_priors[labels[i]]['type'] == "Uniform" or theta_priors[labels[i]]['type'] == "Jeffreys" :
            if not theta_priors[labels[i]]['object'].check_value(theta[i]):
                return -np.inf
            total_prior += theta_priors[labels[i]]['object'].get_ln_prior()
    return total_prior


#make a pairs plot from MCMC output
def pairs_plot(samples, labels, output='') :
    fig = corner.corner(samples, labels = labels)
    plt.show()
    if output != '' :
        fig.savefig(output)   # save the figure to file
        plt.close(fig)


#- Derive best-fit params and their 1-sigm a error bars
def best_fit_params(params, free_param_labels, samples, verbose = False) :
    
    func = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
    
    percents = np.percentile(samples, [16, 50, 84], axis=0)

    seq = list(zip(*percents))
    
    values = list(map(func, seq))
    
    theta, theta_labels, theta_err = [], [], []
    
    for i in range(len(values)) :
        if free_param_labels[i] in params.keys() :
            theta.append(values[i][0])
            theta_err.append((values[i][1],values[i][2]))
            theta_labels.append(free_param_labels[i])
            
            if verbose :
                print(free_param_labels[i], "=", values[i][0], "+", values[i][1],"-", values[i][2])

    params = updateParams(params, theta, theta_labels)

    return params, theta, theta_labels, theta_err


#plot model and data
def plot_individual_datasets(bjd, rvs, rverrs, i, planet_params, calib_params, samples, labels, bjd_limits=[], detach_calib=False) :

    plt.subplot(211)
    
    median_rv = np.median(rvs[i])
    
    if bjd_limits == [] :
        calib = calib_model(len(bjd), i, calib_params, bjd[i])
        rvcurve = rv_model(planet_params, bjd[i])
        if detach_calib :
            modelrv = rvcurve
            plt.plot(bjd[i], calib - median_rv, "--", label='Calib - {0:.3f}'.format(median_rv), lw=0.6)
        else :
            modelrv = calib + rvcurve
        plt.plot(bjd[i], modelrv, label='Model', lw=2)
    else :
        tstep = (bjd[i][-1] - bjd[i][0]) / len(bjd[i])
        bjd_new = np.arange(bjd_limits[0], bjd_limits[1], tstep)
        calib = calib_model(len(bjd), i, calib_params, bjd_new)
        rvcurve = rv_model(planet_params, bjd_new)
        if detach_calib :
            modelrv = rvcurve
            plt.plot(bjd_new, calib - median_rv, "--", label='Calib - {0:.3f}'.format(median_rv), lw=0.6)
        else :
            modelrv = calib + rvcurve
        plt.plot(bjd_new, modelrv, label='Model', lw=2)

    if detach_calib :
        calib_loc = calib_model(len(bjd), i, calib_params, bjd[i])
        plt.errorbar(bjd[i], rvs[i]-calib_loc, yerr=rverrs[i], label='Observations', lw=0.6, fmt='o', ms=2, drawstyle='default')
    else :
        plt.errorbar(bjd[i], rvs[i], yerr=rverrs[i], label='Observations', lw=0.6, fmt='o', ms=2, drawstyle='default')
    
    for theta in samples[np.random.randint(len(samples), size=100)]:
        calib_params = updateParams(calib_params, theta, labels)
        planet_params = updateParams(planet_params, theta, labels)
        calib = calib_model(len(bjd), i, calib_params, bjd[i])
        rvcurve = rv_model(planet_params, bjd[i])
        if detach_calib :
            modelrv = rvcurve
        else :
            modelrv = calib + rvcurve
        plt.plot(bjd[i], modelrv, color='red', lw=0.2, alpha=0.1)
    
    plt.xlabel('BJD')
    plt.ylabel('Radial Velocity (km/s)')

    if bjd_limits == [] :
        plt.xlim((bjd[i][0],bjd[i][-1]))
    else :
        plt.xlim((bjd_limits[0],bjd_limits[1]))

    plt.legend()

    titlestr = 'Radial velocities for dataset # {0}'.format(i)
    plt.title(titlestr)
    plt.grid(True)

    plt.subplot(212)
    resids = rvs[i] - (calib + rvcurve)
    plt.errorbar(bjd[i], resids, yerr=rverrs[i], label='Residuals', lw=0.6, fmt='o', ms=2, drawstyle='default')
    plt.xlabel('BJD')
    plt.ylabel('Residuals')
    if bjd_limits == [] :
        plt.xlim((bjd[i][0],bjd[i][-1]))
    else :
        plt.xlim((bjd_limits[0],bjd_limits[1]))
    plt.legend()

    plt.show()


def analysis_of_residuals(bjd, rvs, rverrs, planet_params, calib_params, output="") :
    
    plt_colors = ['orange','olive', 'brown', 'red', 'purple', 'cyan', 'pink', 'gray', 'blue', 'green']
    
    residuals = []
    
    for i in range(len(bjd)) :
        calib = calib_model(len(bjd), i, calib_params, bjd[i])
        rvcurve = rv_model(planet_params, bjd[i])
        modelrv = calib + rvcurve
        residuals.append(rvs[i] - modelrv)

    global_residuals = []
    for i in range(len(residuals)) :
        global_residuals = np.append(global_residuals,residuals[i])

    fig1 = plt.figure()
    ax=fig1.add_axes((.1,.1,.8,.8))
    plt.grid(color='gray', linestyle='dotted', linewidth=0.4)
    plt.xlim([-0.030,0.030])
    binwidth = 0.003
    binBoundaries = np.arange(min(global_residuals), max(global_residuals) + binwidth, binwidth)

    textstr1 = ""
    for i in range(len(residuals)) :
        textstr1 += plot_histogram_of_residuals(residuals[i], binBoundaries, datasetlabel="Dataset {0}".format(i), color=plt_colors[i])

    textstr = plot_histogram_of_residuals(global_residuals, binBoundaries, datasetlabel="ALL datasets", fill=True) + textstr1

    print(textstr1)

    plt.text(-0.03, 0.20, textstr, fontsize=10, bbox=dict(facecolor='none', edgecolor='gray'))

    plt.legend(fontsize=12)
    plt.xlabel('Residuals (km/s)',fontsize=14)
    plt.ylabel('Probability',fontsize=14)


    if output != "":
        fig1.savefig(output, facecolor='white')   # save the figure to file
    else :
        plt.show()
    plt.close(fig1)


def plot_histogram_of_residuals(residuals, binBoundaries, datasetlabel='', color="blue", fill=False) :
    weights = np.ones_like(residuals)/float(len(residuals))
        
    mu = np.mean(residuals)
    sigma = np.std(residuals)
        
    # the histogram of the data
    n, bins, patches = plt.hist(residuals, bins=binBoundaries, weights=weights, histtype='step', align='mid', facecolor=color, alpha=0.5, fill=fill)
        
    # add a 'best fit' line
    y = stats.norm.pdf(bins, mu, sigma)
    ynorm = y / np.sum(y)
        
    normmodel = plt.plot(bins, ynorm, '--', color=color, label=datasetlabel)

    textstr1 = '{0}: mean = {1:.2f} m/s, sig = {2:.2f} m/s\n'.format(datasetlabel, mu*1000.,sigma*1000.)

    return textstr1
