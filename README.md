# pyRM
Package to fit classical Rossiter-McLaughlin effect using MCMC.


Simple usage example:

```
python ./rm_fit.py --pattern=data/HD189733_t2-K2mask-filtered_ccf.rdb --exoplanet_priors=priors/HD189733.pars -v
```

The following input options are available:
```
--pattern for input data pattern (e.g., --pattern=*rdb.fits).
--exoplanet_priors to input file containing exoplanet priors (e.g., --exoplanet_priors=HD189733.pars)
--calib_priors to input file containing calibration priors (e.g., --calib_priors=calib.pars)
#-p for plotting
-v for verbose
```
