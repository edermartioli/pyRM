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
from copy import deepcopy
import matplotlib
import csv
import priorslib
import glob

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

def RManomaly(dvrad,lbda,Vs,aratio,i,Rratio,omega,eps, e, t, T0, tp):

    #computes anovraie relative to center of transit time
    phase0=((t-T0)%tp)/tp
    phase=np.where(np.less(phase0,0),phase0+1,phase0)

    anomoy=2.*np.pi*phase
    anoexc=anomoy*1.

    anoexc1=anoexc+(anomoy+e*np.sin(anoexc)-anoexc)/(1.-e*np.cos(anoexc))
    while np.max(np.abs(anoexc1-anoexc)) > 1.e-8:
        anoexc=anoexc1*1.
        anoexc1=anoexc+(anomoy+e*np.sin(anoexc)-anoexc)/(1.-e*np.cos(anoexc))

    anovraie=2.*np.arctan(np.sqrt((1.+e)/(1.-e))*np.tan(anoexc/2.))
    
    #computes RM anomaly relative to the anovraie
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
        if (R[j]>ind1) and (R[j]<ind2) and dvrad[j] < 0.:
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
        if (R[j]<ind1) and dvrad[j] < 0. :
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
    ncoefs = (len(params)-n)// n
    
    coefs = []
    relative_bjd = []
    to_id = 'd{0:02d}tc'.format(i)
    tc = params[to_id]
    
    for c in range(int(ncoefs)):
        coeff_id = 'd{0:02d}c{1:1d}'.format(i,c)
        coefs.append(params[coeff_id])
    
    for j in range(len(bjd)):
            relative_bjd.append(bjd[j]-tc)
    p = np.poly1d(np.flip(coefs))
    out_model = p(relative_bjd)
    return out_model

def rv_model(planet_params, bjd) :
    
    per = planet_params['per']
    tau = planet_params['tau']
    phi0 = planet_params['phi0']
    k = planet_params['k']
    omega = planet_params['omega'] * np.pi / 180.
    ecc = planet_params['ecc']
    rv0 = planet_params['rv0']
    lambdap = planet_params['lambda'] * np.pi / 180.
    vsini = planet_params['vsini']
    a_R = planet_params['a_R']
    inc = planet_params['inc'] * np.pi / 180.
    r_R = planet_params['r_R']
    omega_rm = planet_params['omega_rm'] * np.pi / 180.
    ldc = planet_params['ldc']
    
    model = []

    keplerian = vradfction(bjd, k, per, omega, phi0, rv0, ecc)
    vrad = keplerian[0]

    anovraie = keplerian[2]

    dvrad = np.concatenate((np.array([vrad[1]-vrad[0]]),vrad[1:]-vrad[:-1]))
    
    rm_effect = RManomaly(dvrad, lambdap, vsini, a_R, inc, r_R, omega_rm, ldc, ecc, bjd, tau, per)

    return (vrad + rm_effect)


#posterior probability
def lnprob(theta, theta_priors, labels, calib_params, planet_params, bjd, rvs, rverrs):
    #lp = lnprior(theta)
    lp = lnprior(theta_priors, theta, labels)
    if not np.isfinite(lp):
        return -np.inf
    prob = lp + lnlike(theta, theta_priors, labels, calib_params, planet_params, bjd, rvs, rverrs)
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
def lnlike(theta, theta_priors, labels, calib_params, planet_params, bjd, rvs, rverrs):
    prior_planet_params = deepcopy(planet_params)
    prior_calib_params = deepcopy(calib_params)
    
    planet_params = updateParams(planet_params, theta, labels)
    calib_params = updateParams(calib_params, theta, labels)

    sum_of_residuals = 0
    for i in range(len(bjd)) :
        calib = calib_model(len(bjd), i, calib_params, bjd[i])
        rvcurve = calib + rv_model(planet_params, bjd[i])
        residuals = rvs[i] - rvcurve
        for key in prior_planet_params.keys() :
            if ("_err" not in key) and ("_pdf" not in key) :
                pdf_key = "{0}_pdf".format(key)
                if prior_planet_params[pdf_key] == "Normal" :
                    error_key = "{0}_err".format(key)
                    error = prior_planet_params[error_key][1]
                    param_chi2 = ((planet_params[key] - prior_planet_params[key])/error)**2
                    sum_of_residuals += param_chi2
    
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
        if theta_priors[labels[i]]['type'] == "Uniform" or theta_priors[labels[i]]['type'] == "Jeffreys" or theta_priors[labels[i]]['type'] == "Normal_positive" :
            if not theta_priors[labels[i]]['object'].check_value(theta[i]):
                return -np.inf
        total_prior += theta_priors[labels[i]]['object'].get_ln_prior()
    return total_prior


#make a pairs plot from MCMC output
def pairs_plot(samples, labels, calib_params, planet_params,fmt, bn = '', p = False, k = False, od = '', output='', addlabels=True) :
    truths=[]
    font = {'size': 15}
    matplotlib.rc('font', **font)

    newlabels = []
    for lab in labels :
        if lab in calib_params.keys():
            truths.append(calib_params[lab])
        elif lab in planet_params.keys():
            truths.append(planet_params[lab])

        if lab == "vsini":
            newlabels.append(r"v$_{e}$sin(i) [km/s]")
        elif lab == "r_R":
            newlabels.append(r"R$_{p}$/R$_{\star}$")
        elif lab == "lambda":
            newlabels.append(r"$\lambda$ [$^{\circ}$]")
        elif lab == "tau":
            newlabels.append(r"T$_{c}$ [d]")
        elif lab == "d00c1":
            newlabels.append(r"$\gamma$ [km/s]")
        elif lab == "d00c0":
            newlabels.append(r"$\alpha$ [km/s/d]")
        else :
            newlabels.append(lab)
    
    if addlabels :
        fig = corner.corner(samples, labels=newlabels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=truths)
    else :
        fig = corner.corner(samples, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84],truths=truths)
    if k :
        plt.savefig(od+bn+"_pairsplot."+fmt, format = fmt)
    if p :
        plt.show()
    if output != '' :
        fig.savefig(output)
    plt.close(fig)



#- Derive best-fit params and their 1-sigm a error bars
def best_fit_params(params, free_param_labels, samples, use_mean=False, verbose = False) :

    theta, theta_labels, theta_err = [], [], []
    
    if use_mean :
        npsamples = np.array(samples)
        values = []
        for i in range(len(samples[0])) :
            mean = np.mean(npsamples[:,i])
            err = np.std(npsamples[:,i])
            values.append([mean,err,err])
    else :
        func = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        percents = np.percentile(samples, [16, 50, 84], axis=0)
        seq = list(zip(*percents))
        values = list(map(func, seq))

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
def plot_individual_datasets(bjd, rvs, rverrs, i, input_planet_params, input_calib_params, samples, labels, theta_priors, fmt, inf='', p = False, k = False, od='', bjd_limits=[], detach_calib=False) :

    plt.subplot(211)
    
    median_rv = np.median(rvs[i])
    
    calib_params = deepcopy(input_calib_params)
    planet_params = deepcopy(input_planet_params)
    
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
    #plt.grid(True)
    inputdata = sorted(glob.glob(inf))
    inf = ""
    for char in inputdata[i].rsplit('/')[-1]:
        if char.isalnum():
            inf += char
        if char in ["_","-"]:
            inf += char
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
    
    if k :
        plt.savefig(od+inf+"_plot_RV"+"."+fmt, format = fmt)
    if p:
        plt.show()


def analysis_of_residuals(bjd, rvs, rverrs, planet_params, calib_params, theta_priors,bn, fmt, inf='', p = False, k = False, r = False, od='', output="") :
    
    plt_colors = ['orange','olive', 'brown', 'red', 'purple', 'cyan', 'pink', 'gray', 'blue', 'green']
    
    residuals = []
    res_in_transit = []
    res_off_transit = []
    
    tt = transit_duration(planet_params)
    ti = - tt/2
    te = tt/2
    
    
    for i in range(len(bjd)) :
        calib = calib_model(len(bjd), i, calib_params, bjd[i])
        rvcurve = rv_model(planet_params, bjd[i])
        modelrv = calib + rvcurve
        residuals.append(rvs[i] - modelrv)
        res_in_transit_i = []
        res_off_transit_i = []
        for j in range(len(bjd[i])):
            #calculate predicted center of transit for the data set j
            epoch = round((bjd[i][j] - planet_params['tau']) / (planet_params['per']))
            tc = + planet_params['tau'] + epoch * planet_params['per']
            if ((bjd[i][j]-tc < ti) or (bjd[i][j]-tc > te)) :
                res_off_transit_i.append(rvs[i][j] - modelrv[j])
            else :
                res_in_transit_i.append(rvs[i][j] - modelrv[j])
        res_in_transit.append(res_in_transit_i)
        res_off_transit.append(res_off_transit_i)

    global_residuals = []
    
    if r:
        #writing a file with all the adjusted data
        inputdata = sorted(glob.glob(inf))
        for i in range(len(bjd)):
            f = open(inputdata[i],'r')
            inf = ""
            for char in inputdata[i].rsplit('/')[-1]:
                if char.isalnum():
                    inf += char
                if char in ["_","-"]:
                    inf += char
            g = open(od+inf+'_output.txt','w')
            g.truncate(0)
            writer = csv.writer(g, delimiter = "\t")
            
            calib = calib_model(len(bjd), i, calib_params, bjd[i])
            rvcurve = rv_model(planet_params, bjd[i])
            modelrv = calib + rvcurve
            
            per = planet_params['per']
            tau = planet_params['tau']
            phi0 = planet_params['phi0']
            k = planet_params['k']
            omega = planet_params['omega'] * np.pi / 180.
            ecc = planet_params['ecc']
            rv0 = planet_params['rv0']
            keplerian = vradfction(bjd[i], k, per, omega, phi0, rv0, ecc)
            l = f.readline().split()
            l += ['model','kepler','RM','calib','residual']
            writer.writerow(l)
            for k in range(len(bjd[i])):
                l = f.readline().split()
                if (not("---" in l[0])):
                    l.append(modelrv[k])
                    l.append(keplerian[0][k])
                    l.append(rvcurve[k]-keplerian[0][k])
                    l.append(calib[k])
                    l.append(residuals[i][k])
                    writer.writerow(l)
            f.close()
            g.close()

    for i in range(len(residuals)) :
        global_residuals = np.append(global_residuals,residuals[i])

    fig1 = plt.figure()
    ax=fig1.add_axes((.1,.1,.8,.8))
    #plt.grid(color='gray', linestyle='dotted', linewidth=0.4)
    plt.xlim([-0.030,0.030])
    binwidth = 0.003
    binBoundaries = np.arange(min(global_residuals), max(global_residuals) + binwidth, binwidth)

    textstr1 = ""
    for i in range(len(residuals)) :
        textstr1 += plot_histogram_of_residuals(residuals[i], binBoundaries, datasetlabel="Dataset {0}".format(i), color=plt_colors[i])

    textstr = plot_histogram_of_residuals(global_residuals, binBoundaries, datasetlabel="ALL datasets", fill=True) + textstr1
    
    h = open(od+"Parameters output.txt",'a')
    h.write("normal_calibrated models :\n")
    
    print(textstr1)
    

    plt.text(-0.03, 0.20, textstr, fontsize=10, bbox=dict(facecolor='none', edgecolor='gray'))

    plt.legend(fontsize=12)
    plt.xlabel('Residuals (km/s)',fontsize=14)
    plt.ylabel('Probability',fontsize=14)


    if output != "":
        fig1.savefig(output, facecolor='white')   # save the figure to file
    else :
        if k:
            plt.savefig(od+bn+"_plot_residuals."+fmt, format = fmt)
        if p:
            plt.show()
    plt.close(fig1)
    
    def avg(L):
        if(len(L))!= 0:
            return (sum(L) / len(L))
        else:
            print(str(L)+" is empty")
            return
    
    def std(L):
        if len(L) == 0:
            print(str(L)+" is empty")
            return
        else:
            L_V = sum([((x - avg(L)) ** 2) for x in L]) / len(L)
            return (L_V ** 0.5)
    
    chi2 = 0
    print("Parameters mean :")
    if (avg(global_residuals) is float):
        print("mean : "+ str(round(avg(global_residuals)*1e3,2))+" m/s , sigma : "+ str(round(std(global_residuals)*1e3,2))+" m/s")
        h.write("Parameters mean :\n"+"mean : "+ str(round(avg(global_residuals)*1e3,2))+" m/s , sigma : "+ str(round(std(global_residuals)*1e3,2))+" m/s\n")
    for i in range(len(residuals)):
        if(std(res_in_transit[i]) is float):
            print("std of dataset "+str(i)+" during transit : "+str(round(std(res_in_transit[i])*1e3,2))+" m/s")
            h.write("std of dataset "+str(i)+" during transit : "+str(round(std(res_in_transit[i])*1e3,2))+" m/s\n")
        if(std(res_off_transit[i]) is float):
            print("std of dataset "+str(i)+" out of transit : "+str(round(std(res_off_transit[i])*1e3,2))+" m/s")
            h.write("std of dataset "+str(i)+" out of transit : "+str(round(std(res_off_transit[i])*1e3,2))+" m/s\n")
        for j in range(len(residuals[i])):
            chi2 += (residuals[i][j]**2)/(rverrs[i][j]**2)
        
    print('chi2 = '+str(chi2))
    h.write('chi2 = '+str(chi2))
    h.close()

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


def transit_duration(planet_params):

    period = planet_params['per']
    rp_over_rs = planet_params['r_R']
    sma_over_rs = planet_params['a_R']
    inclination = planet_params['inc']
    eccentricity = planet_params['ecc']
    periastron = planet_params['omega']

    ww = periastron * np.pi / 180
    ii = inclination * np.pi / 180
    ee = eccentricity
    aa = sma_over_rs
    ro_pt = (1 - ee ** 2) / (1 + ee * np.sin(ww))
    b_pt = aa * ro_pt * np.cos(ii)
    if b_pt > 1:
        b_pt = 0.5
    s_ps = 1.0 + rp_over_rs
    df = np.arcsin(np.sqrt((s_ps ** 2 - b_pt ** 2) / ((aa ** 2) * (ro_pt ** 2) - b_pt ** 2)))
    abs_value = (period * (ro_pt ** 2)) / (np.pi * np.sqrt(1 - ee ** 2)) * df
    
    return abs_value


def plot_all_datasets(bjd, rvs, rverrs, input_planet_params, input_calib_params, samples, labels, theta_priors, dt_before, dt_after, fmt, res, bn='', p = False, k = False, od = '') :
    
    font = {'size': 16}
    matplotlib.rc('font', **font)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    min_time = 1e60
    max_time = -1e60

    calib_params = deepcopy(input_calib_params)
    planet_params = deepcopy(input_planet_params)

    ref_tc = 0
    
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:olive","darkblue","teal", "indigo", "orangered", "red", "blue", "green", "grey"]
    
    for i in range(len(bjd)) :
        if i > 14 :
            color = [i/len(bjd),1-i/len(bjd),1-i/len(bjd)]
        else :
            color = colors[i]
        #calculate predicted center of transit for the first data set
        epoch = round((bjd[i][0] - planet_params['tau']) / (planet_params['per']))
        tc = + planet_params['tau'] + epoch * planet_params['per']
        if i == 0:
            ref_tc = tc
        calib = calib_model(len(bjd), i, calib_params, bjd[i])
        modelrv = rv_model(planet_params, bjd[i])

        time_from_center = bjd[i] - tc
        if np.min(time_from_center) < min_time :
            min_time = np.min(time_from_center)
        if np.max(time_from_center) > max_time :
            max_time = np.max(time_from_center)

        #ax1.errorbar(time_from_center, rvs[i]-calib, yerr=rverrs[i], lw=0.7, fmt='o', color='k', ms=5, drawstyle='default', alpha=0.8)
        ax1.errorbar(time_from_center, rvs[i]-calib, yerr=rverrs[i], lw=0.7, fmt='o', color=color, ms=5, drawstyle='default', alpha=0.8, label=r"T$_{0}$={1:.4f} BJD".format(i,tc))
    copy_planet_params = deepcopy(planet_params)
    
    if res == "vh" :
        time_step = 1 / (60 * 60 * 48) # 1/2 second in unit of days
    elif res == "h":
        time_step = 1 / (60 * 60 * 24) # 1 second in unit of days
    elif res == "m":
        time_step = 1 / (60 * 48) # 30 seconds in unit of days
    elif res == "l":
        time_step = 1 / (30 * 24) # 2 minutes in unit of days
    elif res == "vl" :
        time_step = 1 / (12 * 24) # 5 minutes in unit of days
    elif res == "el" :
        time_step = 1 / (48) # 30 minutes in unit of days
    else :
        time_step = 1 / (60 * 48) # 30 seconds in unit of days
    model_time = np.arange(min_time - dt_before, max_time+time_step + dt_after, time_step)
    model_bjd = deepcopy(model_time) + ref_tc

    for theta in samples[np.random.randint(len(samples), size=100)]:
        copy_planet_params = updateParams(copy_planet_params, theta, labels)
        modelrv = rv_model(copy_planet_params, model_bjd)
        ax1.plot(model_time, modelrv, color='red', lw=0.2, alpha=0.2)

    final_model = rv_model(planet_params, model_bjd)

    ax1.plot(model_time, final_model, label='Fit model', color="green", lw=2)

    tt = transit_duration(planet_params)
    ti = - tt/2
    te = tt/2

    ax1.axvline(x=ti, ls="--", linewidth=0.7, color='grey', alpha=0.7)
    ax1.axvline(x=0., ls=":", linewidth=0.7, color='grey', alpha=0.7)
    ax1.axvline(x=te, ls="--", linewidth=0.7, color='grey', alpha=0.7)

    #plt.xlabel(r"Time from center of transit [days]")
    ax1.set_ylabel(r"Radial Velocity [km/s]")
    ax1.legend(fontsize=12)

    for i in range(len(bjd)) :
        if i > 14 :
            color = [i/len(bjd),1-i/len(bjd),1-i/len(bjd)]
        else :
            color = colors[i]
        #calculate predicted center of transit for the first data set
        epoch = round((bjd[i][0] - planet_params['tau']) / (planet_params['per']))
        tc = planet_params['tau'] + epoch * planet_params['per']
        calib = calib_model(len(bjd), i, calib_params, bjd[i])
        modelrv = rv_model(planet_params, bjd[i])

        resids = rvs[i] - (calib + modelrv)
        #plt.errorbar(time_from_center, resids, yerr=rverrs[i], lw=0.6, fmt='o', ms=2, drawstyle='default')

        time = bjd[i] - tc
        
        #ax2.errorbar(time, resids, yerr=rverrs[i], lw=0.7, fmt='o', color='k', ms=5, drawstyle='default', alpha=0.8)
        ax2.errorbar(time, resids, yerr=rverrs[i], lw=0.7, fmt='o', color=color, ms=5, drawstyle='default', alpha=0.8)

    ax2.axvline(x=ti, ls="--", linewidth=0.7, color='grey', alpha=0.7)
    ax2.axvline(x=0., ls=":", linewidth=0.7, color='grey', alpha=0.7)
    ax2.axvline(x=te, ls="--", linewidth=0.7, color='grey', alpha=0.7)

    ax2.set_xlabel(r"Time from center of transit [days]")
    matplotlib.rc('font', **font)
    ax2.set_ylabel(r"Residuals [km/s]")

    if k :
        plt.savefig(od+bn+"_alldatasets."+fmt, format = fmt)
       
    if p :
        plt.show()
