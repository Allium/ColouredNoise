me0 = "test_limb"

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
import os, glob
from sys import argv
from LE_Utils import filename_par

from LE_BulkConst import bulk_const

fsa, fsl, fst = 16, 14, 16

"""
Assume S=0
"""

##=============================================================================

def get_limb(r, c1, R):
	"""
	Find the positing and index of the limb.
	"""
	
	Rind = np.abs(R-r).argmin()
	
	c1m = c1[150:Rind/2].mean()
	c1s = c1[150:Rind/2].std()
	
	c1[:150] = c1m
	
	limb = r[c1 > c1m+5*c1s].min()
	
	return limb

##=============================================================================

def plot_bulkconst(histfile, noshow=False):
	"""
	Plot bulk constant with limb onset indicated.
	"""
	me = me0+".plot_bulkconst: "
	
	zoom = False
	
	a = filename_par(histfile,"_a")
	R = filename_par(histfile,"_R")
	assert filename_par(histfile,"_S") == 0.0

	r, Q, er2E, c1 = bulk_const(histfile)[:4]

	c1 /= c1.mean()

	limb = get_limb(r, c1, R)
	
	fig = plt.figure(); ax = fig.gca()
	
	ax.plot(r, Q/Q.mean(), "b-", label=r"$Q(r)$")
	ax.plot(r, er2E/er2E.mean(), "g-", label=r"$\langle\eta^2\rangle(r)$")
	ax.plot(r, c1, "r-", label=r"$Q\langle\eta^2\rangle$")
	
	ax.axvline(limb,c="k",lw=2)
	ax.axvline(R,c="k",lw=2)
	ax.axvspan(0,R,color="yellow",alpha=0.2)
	ax.axvspan(limb,R,color="red",alpha=0.1)
	
	if zoom:
		ax.set_xlim(np.floor(limb),R)
		filesuf = "_zoom"
	else:
		filesuf = ""
	
	ax.set_xlabel(r"$r$", fontsize=fsa)
	ax.set_ylabel(r"Rescaled variable", fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper left",fontsize=fsl)
	fig.suptitle(r"$a = %.1f$, $R = %.1f$"%(a,R), fontsize=fst)
	
	plotfile = os.path.dirname(histfile)+"/LIMB"+os.path.basename(histfile)[4:-4]+filesuf+".jpg"
	fig.savefig(plotfile)
	
	if not noshow:	plt.show()
	
	return
	
##=============================================================================

def plot_limb_a(histdir, noshow=False):
	"""
	Find directories in 
	"""
	me = me0+".plot_limb_a: "
	
	filelist = np.sort(glob.glob(histdir+"/BHIS_*R10.0_S0.0*.npy"))
	assert len(filelist)>1, me+"Check directory."
	
	A = np.zeros(len(filelist))
	L = np.zeros(len(filelist))
	
	for i, histfile in enumerate(filelist):
		
		a = filename_par(histfile,"_a")
		R = filename_par(histfile,"_R")
		assert filename_par(histfile,"_S") == 0.0
		
		r, Q, er2E, c1 = bulk_const(histfile)[:4]
#		c1 = scipy.ndimage.gaussian_filter(c1,1.0,order=0)

		limb = get_limb(r, c1, R)
		
		A[i] = a
		L[i] = R-limb
	
	srtind = np.argsort(A)
	A = A[srtind]
	L = L[srtind]	
	
	fig = plt.figure(); ax = fig.gca()
	
	ax.plot(A, L, "o-", label="Limb")
	
	AA = np.linspace(0,A[-1],100)
	ax.plot(AA, AA**0.5, "--", label=r"$\alpha^{1/2}$")
	
	ax.set_xlabel(r"$\alpha$", fontsize=fsa)
	ax.set_ylabel(r"Limb length", fontsize=fsa)
	fig.suptitle(r"$R=%.1f$, $S=0.0$"%(R), fontsize=fst)
	ax.grid()
	ax.legend(loc="best",fontsize=fsl)
		
	plotfile = histdir+"/LIMB_a.jpg"
	fig.savefig(plotfile)
	
	if not noshow:	plt.show()

	return
	
##=============================================================================	

if os.path.isfile(argv[1]):
	plot_bulkconst(argv[1])
elif os.path.isdir(argv[1]):
	plot_limb_a(argv[1])
elif argv[1] == "all":
	histdir = "Pressure/161019_CIR_DL_dt0.01"
	for histfile in np.sort(glob.glob(histdir+"/BHIS_*R10.0_S0.0*.npy")):
		plot_bulkconst(histfile, True)
	plot_limb_a(histdir, True)
	
