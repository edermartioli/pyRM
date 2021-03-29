# pyRM
Package to fit classical Rossiter-McLaughlin effect using MCMC.


Simple usage example:

```
python ./rm_fit.py --input=data/HD189733_t2-K2mask-filtered_ccf.rdb 
--exoplanet_priors=priors/HD189733.pars --calib_order=1 --nsteps=400 
--walkers=30 --burnin=150 --samples_filename=hd189733_mcmc_samples.h5 -vp
```

The following input options are available:
```
--input to input data pattern (e.g., --input=*rdb.fits).
--exoplanet_priors to input file containing exoplanet priors (e.g., --exoplanet_priors=HD189733.pars)
--calib_priors to input file containing calibration priors (e.g., --calib_priors=calib.pars)
--calib_order to input the order of the polynomial that is added to the model for the calibration of each input dataset
--nsteps to input the number of steps in the MCMC analysis
--walkers to input the number of walkers in the MCMC analysis
--burnin to input the number of burn-in steps in the MCMC analysis
--samples_filename for the file name to save the MCMC samples
-v for verbose
-p for plotting
```
