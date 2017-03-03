
import numpy as np
import scipy as sp
import os, optparse, glob, time
from sys import argv

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

from LE_CSim import force_dlin, force_clin
from LE_Utils import filename_par, fs, set_mplrc

set_mplrc(fs)

"""
Plot Q(x) and Q(r) on top of one another for high R. They should match closely.
"""

try:
	a, R, S = [float(x) for x in argv[1:4]]
except (IndexError, ValueError):
	a = 1.0
	R = 100.0
	S = 95.0

histdirP = "/home/users2/cs3006/Documents/Coloured_Noise/161212_POL_DL_dt0.01_psi--R100/"
histdirC = "/home/users2/cs3006/Documents/Coloured_Noise/161116_CAR_DL_dt0.01--R100/"

histfileP = histdirP+"BHIS_POL_DL_a%.1f_R%.1f_S%.1f_dt0.01.npy"%(a,R,S)
histfileC = histdirC+"BHIS_CAR_DL_a%.1f_R%.1f_S%.1f_dt0.01.npy"%(a,R,S)

HP = np.load(histfileP)
bP = np.load(os.path.dirname(histfileP)+"/BHISBIN"+os.path.basename(histfileP)[4:-4]+".npz")
HC = np.load(histfileC)
bC = np.load(os.path.dirname(histfileC)+"/BHISBIN"+os.path.basename(histfileC)[4:-4]+".npz")

r = 0.5*(bP["rbins"][1:]+bP["rbins"][:-1])
x = 0.5*(bC["xbins"][1:]+bC["xbins"][:-1])

## Spatial density
QP = HP.sum(axis=2).sum(axis=1) / (2*np.pi*r)
QP /= np.trapz(2*np.pi*r*QP, r)
QC = HC.sum(axis=2).sum(axis=1)
QC /= np.trapz(QC, x)

## Potential
if   "_DL_" in histfileP:	fx = force_dlin([x,0],R,S)[0]
elif "_CL_" in histfileP:	fx = force_clin([x,0],R,S,T)[0]
elif "_ML_" in histfileP:	fx = force_mlin([x,0],R,S,T)[0]
elif "_NL_" in histfileP:	fx = force_nlin([x,0],R,S)[0]
else: raise IOError, me+"Force not recognised."
U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()

## ------------------------------------------------------------------------
## 1D
if 1:

	fig, axs = plt.subplots(1,1, figsize=(10,10))

	ax = axs
	ax.plot(r, QP/QP.max(), label=r"$n_{\rm rad}(r)$")
	ax.plot(x, QC/QC.max(), label=r"$n_{\rm car}(x)$")
	
#	ax.plot(x, np.exp(-U)/np.trapz(np.exp(-U),x), "r-", label=r"$Q_{\rm wn}$")
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--",label=r"$U$")
	
	ax.axvspan(S,R, color="yellow",alpha=0.2)
	ax.axvline(R,color="k",lw=1); ax.axvline(S,color="k",lw=1)
	ax.set_xlim((r[0],r[-1]))
	
	ax.xaxis.set_major_locator(NullLocator())
	ax.yaxis.set_major_locator(NullLocator())
	
	ax.set_xlabel(r"$r$ or $x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$n$ (rescaled)", fontsize=fs["fsa"])
	ax.legend(loc=[0.35,0.25]).get_frame().set_alpha(1.0)
	ax.grid()
	
	title = r"Spatial PDFs (rescaled). $\alpha=%.1f, R=%.1f, S=%.1f$"%(a,R,S)				
#	fig.suptitle(title, fontsize=fs["fst"])
	
	plotfile = os.path.dirname(histfileP)+"/PChighR_a%.1f_R%.1f_S%.1f"%(a,R,S)
	plotfile += "."+fs["saveext"]
	fig.savefig(plotfile, format=fs["saveext"])
	print "Figure saved to",plotfile
	
## ------------------------------------------------------------------------
## 2D
if 0:
	pass	
## ------------------------------------------------------------------------

plt.show()
