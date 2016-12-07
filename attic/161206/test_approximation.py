import os, time
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from LE_Utils import filename_par, fs, set_mplrc
from LE_CSim import force_nlin

set_mplrc(fs)

##=============================================================================

def main():
	"""
	Test approximation for perturbed harmonic well in Cartesian geometry.
	"""
	
	histfile = argv[1]

	assert "_NL_" in histfile
	assert "_T" not in histfile

	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")

	## Data
				
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	H = np.load(histfile)
	Q = H.sum(axis=2).sum(axis=1) / (H.sum()*(x[1]-x[0]))

	## Prediction

	xp = np.linspace(-4.0,4.0,1001)
	Qp = calc_Q_NLsmallR(xp,a,R,S)
	Qp /= np.trapz(Qp, xp)

	## Plotting

	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])

	ax.plot(x, Q,  lw=2, label="Simulation")
	ax.plot(xp,Qp, lw=2, label="Prediction")

	ax.set_xlim((x[0],x[-1]))
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$Q(x)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="upper left", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)

	fig.tight_layout()
	fig.subplots_adjust(top=0.90)
	title = r"Approximate Spatial PDF. $\alpha=%.1f, R=%.1f, S=%.1f$"%(a,R,S)
	fig.suptitle(title, fontsize=fs["fst"])

	plotfile = os.path.dirname(histfile)+"/APPR"+os.path.basename(histfile)[4:-4]+".jpg"
	fig.savefig(plotfile)

	plt.show()
	
	return

##=============================================================================

def calc_rho_NLsmallR(x, a, R, S):
	"""
	Calculate 1o solution for the small-R nlin scenario.
	See notes 22/11/2016.
	"""
	Rind, Sind = np.abs(x-R).argmin(), np.abs(x-S).argmin()
	DD = np.zeros(x.size); DD[Sind] = 1.0 / (x[1]-x[0])
	return

def calc_Q_NLsmallR(x, a, R, S):
	"""
	Calculate 1o spatial density for the small-R nlin scenario.
	See notes 22/11/2016.
	"""
	Rind, Sind = np.abs(x-R).argmin(), np.abs(x-S).argmin()
	DD = np.zeros(x.size); DD[Sind] = 1.0 / (x[1]-x[0])
	Q0R = np.sqrt((a+1)/(2*np.pi))*np.exp(-0.5*(a+1)*(x)**2)
	Q0L = np.sqrt((a+1)/(2*np.pi))*np.exp(-0.5*(a+1)*(x)**2)
	n = 1/(1-2*R*np.sqrt((a+1)/(2*np.pi))*np.exp(-0.5*(a+1)*S*S))	## Maybe wrong
	QR = n*(1 + R*(a+1)*x - 2*R*DD)*Q0R
	QL = n*(1 - R*(a+1)*x - 2*R*DD)*Q0L
	Q = np.r_[QL[:Sind],QR[Sind:]]
	return Q

##=============================================================================
##=============================================================================
is __name__=="__main__":
	main()