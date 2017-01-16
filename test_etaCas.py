me0 = "test_etaPDF"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv
import os
from LE_Utils import filename_par, fs, set_mplrc

set_mplrc(fs)

##=================================================================================================

def main():
	"""
	Plot pdf of eta if given a file. pdf split into regions.
	Adapted from test_etaPDF.py
	Can be called as standalone, or plot_peta imported.
	"""
	me = me0+".main: "

	histfile = argv[1]

	## To save, need argv2. 	try: nosave = not bool(argv[2])
	except IndexError: nosave = True
	
	fig, ax = plt.subplots(1,1)
	
	plot_peta_CL(histfile, fig, ax, nosave)
	
	plt.show()
	
	return

##=================================================================================================

def plot_peta_CL(histfile, fig, ax, nosave=True):
	"""
	"""
	me = me0+".plot_peta_CL: "

	assert ("_CL_" in histfile) or ("_ML_" in histfile), me+"Designed for Casmir geometry."

	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	T = filename_par(histfile, "_T")

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
	## 
	if  "_CL_" in histfile:
		qin  = H[Tind]
		qout = np.trapz(H[Sind:Rind+1], x[Sind:Rind+1], axis=0)
	elif "_ML_" in histfile:
		## in is now right region and out is left region
		qin  = np.trapz(H[Sind:Rind+1], x[Sind:Rind+1], axis=0) if Sind!=Rind else H[Sind]
		qout = np.trapz(H[x.size-Rind:Tind], x[x.size-Rind:Tind], axis=0)

	## Normalise each individually so we can se just distrbution
	qin /= np.trapz(qin, ex)
	qout /= np.trapz(qout, ex)
		
	##---------------------------------------------------------------	
	## PDF PLOT

	ax.plot(ex, qin, "b-", label="Interior")
	ax.fill_between(ex,0,qin,facecolor="blue",alpha=0.1)
	ax.plot(ex, qout, "g-", label=r"Bulk")
	ax.fill_between(ex,0,qout,facecolor="green",alpha=0.1)

	## Accoutrements
	ax.set_xlabel(r"$\eta$")
	ax.set_ylabel(r"$p(\eta)$")
	ax.grid()
	
	## Make legend if standalone
	if not nosave:
		ax.legend()

	# fig.suptitle(r"PDF of $\eta$, divided into regions. $\alpha="+str(a)+"$, $R="+str(R)+"$, $S="+str(S)+"$")

	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFeta"+os.path.basename(histfile)[4:-4]+".pdf"
		fig.savefig(plotfile)
		print me+": Figure saved to",plotfile
	
	return


##=================================================================================================

##=================================================================================================
if __name__=="__main__":
	main()
