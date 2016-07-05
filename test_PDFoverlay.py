import numpy as np
from matplotlib import pyplot as plt

import os, time, sys
from LE_Utils import filename_pars
from LE_SPressure import calc_pressure

"""
Reads in BHIS files of different bulk sizes / alphas and overlays wall regions. See figure in document 160627.
"""
R=100.0

## Filename
if float(sys.argv[1])==10.0:
	histfiles = ["Pressure/160619_CIR_DLN_R100.0_Pa/DL_R100.0/BHIS_CIR_DL_a10.0_R100.0_S100.0_dt0.02.npy",\
				"Pressure/160701_CIR_DL_dt0.01_smallbulk/DL_0.1b/BHIS_CIR_DL_a10.0_R100.0_S99.9_dt0.01.npy",\
				# "Pressure/160701_CIR_DL_dt0.01_smallbulk/DL_0.5b/BHIS_CIR_DL_a10.0_R100.0_S99.5_dt0.01.npy",\
				"Pressure/160619_CIR_DLN_R100.0_Pa/DL_R100.0/BHIS_CIR_DL_a10.0_R100.0_S99.0_dt0.02.npy",\
				"Pressure/160619_CIR_DLN_R100.0_Pa/DL_R100.0/BHIS_CIR_DL_a10.0_R100.0_S90.0_dt0.02.npy"]
elif float(sys.argv[1])==0.2:
	histfiles = ["Pressure/160619_CIR_DLN_R100.0_Pa/DL_R100.0/BHIS_CIR_DL_a0.2_R100.0_S100.0_dt0.01.npy",\
				"Pressure/160701_CIR_DL_dt0.01_smallbulk/DL_0.1b/BHIS_CIR_DL_a0.2_R100.0_S99.9_dt0.005.npy",\
				# "Pressure/160701_CIR_DL_dt0.01_smallbulk/DL_0.5b/BHIS_CIR_DL_a0.2_R100.0_S99.5_dt0.005.npy",\
				"Pressure/160619_CIR_DLN_R100.0_Pa/DL_R100.0/BHIS_CIR_DL_a0.2_R100.0_S99.0_dt0.01.npy",\
				"Pressure/160619_CIR_DLN_R100.0_Pa/DL_R100.0/BHIS_CIR_DL_a0.2_R100.0_S90.0_dt0.01.npy"]
else:
	print "argv must be 10 or 0.2 depending on alpha."

			
fig, ax = plt.subplots(1,1,sharex=True)

for histfile in histfiles:

	## Get pars from filename
	pars = filename_pars(histfile)
	[a,ftype,R,S,lam,nu] = [pars[key] for key in ["a","ftype","R","S","lam","nu"]]
	fpars = [R,S,lam,nu]
		
	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	rmax = rbins[-1]
	r = 0.5*(rbins[1:]+rbins[:-1])
	erbins = bins["erbins"]
	er = 0.5*(erbins[1:]+erbins[-1])
	rini = 0.5*(max(rbins[0],S)+R)	## Start point for computing pressures
	rinid = np.argmin(np.abs(r-rini))

	## Load histogram, convert to normalised pdf
	H = np.load(histfile)
	try:	H = H.sum(axis=2)
	except ValueError:	pass
	## Noise dimension irrelevant here
	H = np.trapz(H, x=er, axis=1)
	## rho is probability density. H is probability at r
	rho = H/(2*np.pi*r) / np.trapz(H, x=r, axis=0)
	rho/=rho.max()
		
	##---------------------------------------------------------------	
	## PDF PLOT

	## PDF and WN PDF
	ax.plot(r,rho, label="CN simulation", zorder=1)

ax.plot(r,np.exp(-0.5*(r-R)*(r-R)),"--")
	
# ax.axvspan(96.0,100.0,facecolor="k",alpha=0.3, zorder=2)
# ax.axvline(100.0, c="k")
	
ax.set_xlim([100.0,104.0])
ax.grid()
ax.legend(["$R-S=0.0$","$R-S=0.1$","$R-S=1.0$","$R-S=10.0$","WN"])
ax.set_xlabel("$r$",fontsize=16)
ax.set_ylabel("$\\rho(r)$ (scaled)",fontsize=16)
ax.set_title("$\\alpha="+str(sys.argv[1])+"$", fontsize=16)
		
plt.show()
