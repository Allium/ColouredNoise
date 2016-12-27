me0 = "test_etaPDF"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv
import os
from LE_Utils import filename_par, fs, set_mplrc

set_mplrc(fs)

"""
Plot pdf of eta if given a file. pdf split into three regions.
Adapted from test_etaPDF.py
"""

histfile = argv[1]

assert "_CL_" in histfile, "Desigend for Casmir geometry."

a = filename_par(histfile, "_a")
R = filename_par(histfile, "_R")
S = filename_par(histfile, "_S")
T = filename_par(histfile, "_T")


## To save, need argv2
try: nosave = not bool(argv[2])
except IndexError: nosave = True

## Space (for axes)
bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
xbins = bins["xbins"]
xmax = xbins[-1]
x = 0.5*(xbins[1:]+xbins[:-1])
exbins = bins["exbins"]
ex = 0.5*(exbins[1:]+exbins[:-1])

## Wall indices
Rind = np.abs(x-R).argmin()
Sind = np.abs(x-S).argmin()
Tind = np.abs(x-T).argmin()
cuspind = np.abs(x-0.5*(S+T)).argmin()
	
## Load histogram; normalise
H = np.load(histfile)
try:	H = H.sum(axis=2)
except ValueError:	pass
H /= np.trapz(np.trapz(H,ex,axis=1),x,axis=0)

## Distribution on either side of the wall: inner, outer
xin = x[:cuspind]
Hin = H[:cuspind,:]
xout = x[cuspind:]
Hout = H[cuspind:,:]

## q is probability density in eta. r is no longer relevant.
# qin  = np.trapz(Hin, xin, axis=0)
qin  = Hin[Tind]
qout = np.trapz(Hout[Sind:Rind+1], xout[Sind:Rind+1], axis=0)

## Normalise each individually so we can se just distrbution
qin /= np.trapz(qin, ex)
qout /= np.trapz(qout, ex)

##---------------------------------------------------------------			
## PLOT SET-UP
	
fig, ax = plt.subplots(1,1)
		
##---------------------------------------------------------------	
## PDF PLOT

ax.plot(ex, qin, "b-", label="Inner")
ax.fill_between(ex,0,qin,facecolor="blue",alpha=0.1)
ax.plot(ex, qout, "g-", label=r"Outer")
ax.fill_between(ex,0,qout,facecolor="green",alpha=0.1)

## Accoutrements
ax.set_xlabel(r"$\eta_x$")
ax.set_ylabel(r"$q(\eta)$")
ax.grid()
ax.legend()

# fig.suptitle(r"PDF of $\eta$, divided into regions. $\alpha="+str(a)+"$, $R="+str(R)+"$, $S="+str(S)+"$")

if not nosave:
	plotfile = os.path.dirname(histfile)+"/PDFeta"+os.path.basename(histfile)[4:-4]+".pdf	"
	fig.savefig(plotfile)
	print me0+": Figure saved to",plotfile
	
# plt.show()
exit()