A complete Heston workflow from calibration to pricing/greeks calculation.

`heston_calibration.py`:  
Heston calibraiton using Gatheral's 2-step technique, i.e. first, fix rho and calibrate the remaining parameters to the full implied volatility surface, primarily targeting the term structure and smile curvature. Then set $\rho$ and $\sigma$ (in usual Heston nomenclature $\xi$) free and give the already-calibrated $\nu_0$, $\theta$ and $\kappa$ a bound for a second calibration to fit the strike curve. I also set an optional $1/T$ weight on the squared IV residuals for the params calibration stage 2, an option to fix $\kappa$ during calibrations and allow a historical params penalty to decrease the fluctuation of params between calibrations. 

`heston_pricing.py`:  
Use ADI scheme to price 3 different commonly traded structures on Chinese market, i.e. snowball(雪球), autocallable(小雪球/保本雪球), phoenix(凤凰). A greeks calculation function is also attached.

`heston_workflow.ipynb`:  
The notebook to work with, integrated calibration and pricing/greeks calculation.

`vol_surface.xlsx`:  
The implied vol surface file downloaded from `wind`. You may replace it with iv surface from any source.# Heston-demo
