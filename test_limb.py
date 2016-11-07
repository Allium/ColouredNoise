me0 = "test_limb"

import numpy as np
import scipy.optimize
import os, glob
from sys import argv

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
from matplotlib import pyplot as plt

from LE_Utils import filename_par, fs

fsa, fsl, fst = fs

"""
Assume S=0
"""

##=============================================================================

def get_limb(r, arr, R):
	"""
	Find the position of the limb.
	"""
	
	Rind = np.abs(R-r).argmin()
	
	m = arr[100:Rind/2].mean()
	s = arr[100:Rind/2].std()
	
	arr[:100] = m
	
	try:
		limb = min( r[arr > m+5*s].min(), r[arr < m-5*s].min() )
	except ValueError:
		## If the first array in min is empty
		limb = r[arr < m-5*s].min()
	
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

	r, Q, e2c2, e2s2 = calculate_arrs(histfile)

	Q /= Q.mean()
	e2c2 /= e2c2.mean()
	e2s2 /= e2s2.mean()

	limbQ = get_limb(r, Q, R)
	limbC = get_limb(r, e2c2, R)
	limbS = get_limb(r, e2s2, R)
	
	fig = plt.figure(); ax = fig.gca()
	
	lineQ = ax.plot(r, Q, label=r"$Q(r)$")
	lineC = ax.plot(r, e2c2, label=r"$\langle\eta^2\cos^2\psi\rangle(r)$")
	lineS = ax.plot(r, e2s2, label=r"$\langle\eta^2\sin^2\psi\rangle(r)$")
	
	ax.axvline(limbQ, c=lineQ[0].get_color(), lw=2)
	ax.axvline(limbC, c=lineC[0].get_color(), lw=2)
	ax.axvline(limbS, c=lineS[0].get_color(), lw=2)
	
	ax.axvline(R,c="k",lw=2)
	ax.axvspan(0,R,color="yellow",alpha=0.2)
	
	if zoom:
		ax.set_xlim(np.floor(limb),R)
		filesuf = "_zoom"
	else:
		filesuf = ""
	
	ax.set_xlabel(r"$r$", fontsize=fsa)
	ax.set_ylabel(r"Rescaled variable", fontsize=fsa)
	ax.set_title(r"$a = %.1f$, $R = %.1f$"%(a,R), fontsize=fst)
	
	ax.grid()
	ax.legend(loc="upper left",fontsize=fsl)
	
	plotfile = os.path.dirname(histfile)+"/LIMB"+os.path.basename(histfile)[4:-4]+filesuf+".jpg"
	fig.savefig(plotfile)
	print me+"Plot saved to",plotfile
	
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
	LQ, LC, LS = np.zeros((3,len(filelist)))
	
	for i, histfile in enumerate(filelist):
		
		A[i] = filename_par(histfile,"_a")
		R = filename_par(histfile,"_R")
		assert filename_par(histfile,"_S") == 0.0
		
		r, Q, e2c2, e2s2 = calculate_arrs(histfile)		
		
		LQ[i] = R - get_limb(r, Q, R)
		LC[i] = R - get_limb(r, e2c2, R)
		LS[i] = R - get_limb(r, e2s2, R)
	
	## --------------------------------------------------------------------	
	
	srtind = np.argsort(A)
	A = A[srtind]
	LQ = ( LQ[srtind] )
	LC = ( LC[srtind] )
	LS = ( LS[srtind] )
	
	## --------------------------------------------------------------------	
	
	fig = plt.figure(); ax = fig.gca()
	
	ax.plot(A, LQ, "o-", label=r"$Q$")
	ax.plot(A, LC, "o-", label=r"$\langle\eta^2 \cos^2\psi\rangle$")
	ax.plot(A, LS, "o-", label=r"$\langle\eta^2 \sin^2\psi\rangle$")
	
	AA = np.linspace(0.0,A[-1],100)
	ax.plot(AA, AA**0.5, "--", label=r"$\alpha^{1/2}$")
	
	ax.set_xlabel(r"$\alpha$", fontsize=fsa)
	ax.set_ylabel(r"Limb length", fontsize=fsa)
	ax.set_title(r"$R=%.1f$, $S=0.0$"%(R), fontsize=fst)
	ax.grid()
	ax.legend(loc="best",fontsize=fsl).get_frame().set_alpha(0.5)
		
	plotfile = histdir+"/LIMB_a.jpg"
	fig.savefig(plotfile)
	print me+"Plot saved to",plotfile
	
	if not noshow:	plt.show()

	return
	
##=============================================================================	
def calculate_arrs(histfile):
	"""
	Calculate arrays to plot.
	"""
		
	H = np.load(histfile)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	
	## Space and load histogram
	rbins = bins["rbins"]
	erbins = bins["erbins"]
	r = 0.5*(rbins[1:]+rbins[:-1])
	etar = 0.5*(erbins[1:]+erbins[:-1])
	epbins = bins["epbins"]
	etap = 0.5*(epbins[1:]+epbins[:-1])
	
	## Spatial arrays with dimensions commensurate to rho
	rr = r[:,np.newaxis,np.newaxis]
	ee = etar[np.newaxis,:,np.newaxis]
	pp = etap[np.newaxis,np.newaxis,:]
	dV = (r[1]-r[0])*(etar[1]-etar[0])*(etap[1]-etap[0])	## Assumes regular grid

	## --------------------------------------------------------------------
	
	## Normalise histogram and convert to density
	H /= H.sum()*dV
	rho = H / ( (2*np.pi)**2.0 * rr*ee )
	
	## Marginalise over eta and calculate BC
	## Radial density
	Q = np.trapz(np.trapz(rho, etap, axis=2)*etar, etar, axis=1) * 2*np.pi
	Qp = Q + (Q==0.0)
	
	## <\eta^2\cos^2\psi>Q, <\eta^2\sin^2\psi>Q
	e2c2Q = np.trapz(np.trapz(rho * np.cos(pp)*np.cos(pp), etap, axis=2)*etar*etar * 2*np.pi*etar, etar, axis=1)
	e2s2Q = np.trapz(np.trapz(rho * np.sin(pp)*np.sin(pp), etap, axis=2)*etar*etar * 2*np.pi*etar, etar, axis=1)
	
	return r, Q, e2c2Q/Qp, e2s2Q/Qp


##=============================================================================	

if os.path.isfile(argv[1]):
	plot_bulkconst(argv[1])
elif (os.path.isdir(argv[1]) and len(argv) > 3 and argv[2] == "all"):
	for histfile in np.sort(glob.glob(argv[1]+"/BHIS_*R10.0_S0.0*.npy")):
		plot_bulkconst(histfile, True)
		plt.close()
	plot_limb_a(argv[1], True)
elif os.path.isdir(argv[1]):
	plot_limb_a(argv[1])
	
