
import numpy as np
import scipy as sp
import os, optparse, glob, time

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt

from LE_CSim import force_dlin, force_clin
from LE_Utils import filename_par
from LE_Utils import fs
fsa,fsl,fst = fs

"""
Plot Q(x) and Q(r) on top of one another for high R. They should match closely.
"""

a = 10.0
R = 100.0
S = 100.0

histdirP = "/home/users2/cs3006/Documents/Coloured_Noise/161119_POL_DL_dt0.01_psi--R100/"
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

## 1D
if 1:

	fig, axs = plt.subplots(1,1, figsize=(10,10))

	ax = axs#[0]
	ax.plot(r, QP/QP.max(), label=r"$Q(r)$")
	ax.plot(x, QC/QC.max(), label=r"$Q(x)$")
	ax.axvline(R,color="k"); ax.axvline(S,color="k")
	ax.set_xlim((r[0],r[-1]))
	ax.set_xlabel(r"$r$ or $x$", fontsize=fsa)
	ax.set_ylabel(r"$Q$", fontsize=fsa)
	ax.legend(fontsize=fsl)
	ax.grid()

	#fig.tight_layout()

plt.show()
