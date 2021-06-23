# -*- coding: iso-8859-1 -*-
"""
    Created on February 3 2020
    
    Description: Priors library
    
    @author: Eder Martioli <martioli@iap.fr>, Shweta Dalal <dalal@iap.fr>, Alexandre Teissier
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np

def get_quantiles(dist, alpha = 0.68, method = 'median'):
    """
    get_quantiles function
    DESCRIPTION
        This function returns, in the default case, the parameter median and the error%
        credibility around it. This assumes you give a non-ordered
        distribution of parameters.
    OUTPUTS
        Median of the parameter,upper credibility bound, lower credibility bound
    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]

class normal_parameter:
      """
      Description
      -----------
      This class defines a parameter object which has a normal prior. It serves
      to save both the prior and the posterior chains for an easier check of the parameter.
      """
      
      def __init__(self,prior_hypp, pos = False):
          self.value = prior_hypp[0]
          self.value_u = 0.0
          self.value_l = 0.0
          self.prior_hypp = prior_hypp
          self.posterior = []
          self.is_positive = pos

      def get_ln_prior(self):
          return np.log(1./np.sqrt(2.*np.pi*(self.prior_hypp[1]**2)))-\
                 0.5*(((self.prior_hypp[0]-self.value)**2/(self.prior_hypp[1]**2)))

      def set_value(self,new_val):
          self.value = new_val

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l
      
      def check_value(self, x):
          return ((self.is_positive and x < 0) == False)

class uniform_parameter:
      """
      Description
      -----------
      This class defines a parameter object which has a uniform prior. It serves
      to save both the prior and the posterior chains for an easier check of the parameter.
      """
      def __init__(self,prior_hypp):
          self.value = (prior_hypp[0]+prior_hypp[1])/2.
          self.value_u = 0.0
          self.value_l = 0.0
          self.prior_hypp = prior_hypp
          self.posterior = []

      def get_ln_prior(self):
          return np.log(1./(self.prior_hypp[1]-self.prior_hypp[0]))

      def check_value(self,x):
          if x > self.prior_hypp[0] and  x < self.prior_hypp[1]:
              return True
          else:
              return False
 
      def set_value(self,new_val):
          self.value = new_val

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l

class jeffreys_parameter:
      """
      Description
      -----------
      This class defines a parameter object which has a Jeffreys prior. It serves
      to save both the prior and the posterior chains for an easier check of the parameter.
      """
      def __init__(self,prior_hypp):
          self.value = np.sqrt(prior_hypp[0]*prior_hypp[1])
          self.value_u = 0.0
          self.value_l = 0.0
          self.prior_hypp = prior_hypp
          self.posterior = []

      def get_ln_prior(self):
          return np.log(1.0) - np.log(self.value*np.log(self.prior_hypp[1]/self.prior_hypp[0]))

      def check_value(self,x):
          if x > self.prior_hypp[0] and  x < self.prior_hypp[1]:
              return True
          else:
              return False

      def set_value(self,new_val):
          self.value = new_val

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l

class constant_parameter:
      """
      Description
      -----------
      This class defines a parameter object which has a constant value. It serves
      to save both the prior and the posterior chains for an easier check of the parameter.
      """
      def __init__(self,val):
          self.value = val

def read_priors(filename, calibration = False):
    def generate_parameter(values):
        out_dict = {}
        out_dict['type'] = values[1]
        if values[1] == 'Normal':
            out_dict['object'] = normal_parameter(np.array(values[2].split(',')).astype('float64'))
        elif values[1] == 'Normal_positive':
            out_dict['object'] = normal_parameter(np.array(values[2].split(',')).astype('float64'), True)
        elif values[1] == 'Uniform':
            out_dict['object'] = uniform_parameter(np.array(values[2].split(',')).astype('float64'))
        elif values[1] == 'Jeffreys':
            out_dict['object'] = jeffreys_parameter(np.array(values[2].split(',')).astype('float64'))
        elif values[1] == 'FIXED':
            out_dict['object'] = constant_parameter(np.array(values[2].split(',')).astype('float64')[0])
        if len(values)>=5:
            out_dict['object'].set_value(np.float(values[3]))
        return out_dict
    # Open the file containing the priors:
    f = open(filename)
    # Generate dictionary that will save the data on the priors:
    priors = {}
    n_coefs = 0
    while True:
        line = f.readline()
        if line == '':
            break
        elif line[0] != '#':
            # Extract values from text file: [0]: parameter name,
            #                                [1]: prior type,
            #                                [2]: hyperparameters,
            #                                [3]: starting value (optional)
            values = line.split()
            priors[values[0]] = generate_parameter(values)
            errors = np.array(values[2].split(',')).astype('float64')
            error_key = "{0}_err".format(values[0])
            pdf_key = "{0}_pdf".format(values[0])
            priors[pdf_key] = values[1]
            priors[error_key] = errors
            n_coefs += 1

    f.close()

    if calibration :
        n_coefs-= 1
        ndatasets = n_coefs // (1+float(priors['orderOfPolynomial']['object'].value))
        priors["ndatasets"] = generate_parameter(['ndatasets','FIXED',str(ndatasets)])
    return priors

def read_exoplanet_params(prior_dict, output_theta_params = False):

    param_ids = ['per','tau','phi0','k','omega','ecc','rv0','lambda', 'vsini','a_R','inc','r_R','omega_rm','ldc']

    planet_params = {}
    
    for i in range(len(param_ids)):
        param = prior_dict[param_ids[i]]
        planet_params[param_ids[i]] = param['object'].value
    
    if output_theta_params :
        theta, labels = [], []
        for key in planet_params.keys():
            param = prior_dict[key]
            if param['type'] != 'FIXED':
                theta.append(planet_params[key])
                labels.append(key)
        return theta, labels
    else :
        for i in range(len(param_ids)):
            error_key = "{0}_err".format(param_ids[i])
            planet_params[error_key] = prior_dict[error_key]
            pdf_key = "{0}_pdf".format(param_ids[i])
            planet_params[pdf_key] = prior_dict[pdf_key]
        return planet_params

def read_calib_params(calib_priors_dict, output_theta_params = False):

    order_of_polynomial = calib_priors_dict['orderOfPolynomial']['object'].value
    ndatasets = calib_priors_dict['ndatasets']['object'].value

    calib_params = {}
    
    for i in range(int(ndatasets)):
        to_id = 'd{0:02d}tc'.format(i)
        calib_params[to_id] = calib_priors_dict[to_id]['object'].value
        for c in range(int(order_of_polynomial)):
            coeff_id = 'd{0:02d}c{1:1d}'.format(i,c)
            calib_params[coeff_id] = calib_priors_dict[coeff_id]['object'].value

    if output_theta_params :
        theta, labels = [], []
        for key in calib_params.keys():
            param = calib_priors_dict[key]
            if param['type'] != 'FIXED':
                theta.append(calib_params[key])
                labels.append(key)
        return theta, labels
    else :
        return calib_params

#intialize calib parameters
def init_calib_priors(tc, ndim=1, order=1, coefs=None) :
    """
        ndim : number of polynomials, typically the number of different datasets
        order:  number of coefficients in the polynomial
        coefs: array of arrays of coefficients.
        
        e.g.: if ndim=2, then coefs=[[3,2],[1,3]]
        """
    priors = {}

    order_dict = {}
    order_dict['type'] = "FIXED"
    order_dict['object'] = constant_parameter(order)
    priors['orderOfPolynomial'] = order_dict
    
    nds_dict = {}
    nds_dict['type'] = "FIXED"
    nds_dict['object'] = constant_parameter(ndim)
    priors['ndatasets'] = nds_dict
    
    for i in range(int(ndim)):
        to_id = 'd{0:02d}tc'.format(i)
        t_dict = {}
        t_dict['type'] = "FIXED"
        t_dict['object'] = constant_parameter(tc)
        priors[to_id] = t_dict
        for c in range(order):
            coeff_id = 'd{0:02d}c{1:1d}'.format(i,c)
            out_dict = {}
            out_dict['type'] = "Uniform"
            l_value = -1e20
            u_value = +1e20
            
            out_dict['object'] = uniform_parameter(np.array([l_value, u_value]))
            
            if coefs != None :
                out_dict['object'].set_value(np.float(coefs[i][c]))

            priors[coeff_id] = out_dict

    return priors


def get_theta_from_priors(planet_priors, calib_priors) :
    
    theta_planet, planet_labels = read_exoplanet_params(planet_priors, output_theta_params = True)

    theta_calib, calib_labels = read_calib_params(calib_priors, output_theta_params = True)

    theta = np.concatenate((theta_planet, theta_calib), axis=0)

    labels = planet_labels + calib_labels

    theta_priors = {}
    for key in planet_priors.keys() :
        if key in labels :
            theta_priors[key] = planet_priors[key]
    for key in calib_priors.keys() :
        if key in labels :
            theta_priors[key] = calib_priors[key]

    return theta, labels, theta_priors


def save_posterior(output, params, theta_fit, theta_labels, theta_err, calib=False, ncoeff=1, ref_time=0.) :

    outfile = open(output,"w")
    outfile.write("# Parameter_ID\tPrior_Type\tValues\n")
    if calib :
        outfile.write("orderOfPolynomial\tFIXED\t{0}\n".format(ncoeff))
    
    for key in params.keys() :
        if key in theta_labels :
            idx = theta_labels.index(key)
            error = (theta_err[idx][0] + theta_err[idx][1]) / 2.
            if key == 'tau' :
                outfile.write("{0}\tNormal\t{1:.4f},{2:.4f}\n".format(key, theta_fit[idx]+ref_time, error))
            else :
                outfile.write("{0}\tNormal\t{1:.4f},{2:.4f}\n".format(key, theta_fit[idx], error))
        else :
            if key == 'tau' :
                outfile.write("{0}\tFIXED\t{1}\n".format(key,params[key]+ref_time))
            else :
                outfile.write("{0}\tFIXED\t{1}\n".format(key,params[key]))

    outfile.close()
