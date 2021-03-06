me0 = "test_etaCas"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import os, optparse, glob, time
from LE_Utils import filename_par, fs, set_mplrc

from LE_CSim import force_mlin

set_mplrc(fs)

##=================================================================================================

def main():
	"""
	Plot pdf of eta if given a file. pdf split into regions.
	Adapted from test_etaPDF.py
	Can be called as standalone, or plot_peta imported.
	"""
	me = me0+".main: "

	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('--str',
		dest="searchstr", default="", type="str")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	searchstr = opt.searchstr
	nosave = opt.nosave
	vb = opt.verbose

	histfile = args[0]
	
	fig, ax = plt.subplots(1,1)
	
	plot_peta_CL(histfile, fig, ax, nosave)
	
	if showfig:
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
		labels = ["Interior","Bulk"]
		colour = ["g","b"]
	elif "_ML_" in histfile:
		## in is now right region and out is left region
		qin  = np.trapz(H[Sind:Rind+1], x[Sind:Rind+1], axis=0) if Sind!=Rind else H[Sind]
		qout = np.trapz(H[x.size-Rind:Tind], x[x.size-Rind:Tind], axis=0)
		labels = ["Small bulk","Large bulk"]
		colour = ["b","g"]

	## Normalise each individually so we can see just distrbution
	qin /= np.trapz(qin, ex)
	qout /= np.trapz(qout, ex)
		
	##---------------------------------------------------------------	
	## PDF PLOT
	
	ax.plot(ex, qout, colour[0]+"-", label=labels[1])
	ax.fill_between(ex,0,qout,facecolor=colour[0],alpha=0.1)
	ax.plot(ex, qin, colour[1]+"-", label=labels[0])
	ax.fill_between(ex,0,qin,facecolor=colour[1],alpha=0.1)
	
##	##---------------------------------------------------------------	
##	## Entire in/out region
##	qIN  = np.trapz(H[0:cuspind], x[0:cuspind], axis=0)
##	qOUT = np.trapz(H[cuspind:], x[cuspind:], axis=0)
##	## Normalise pdf
###	qIN /= np.trapz(qIN, ex)
###	qOUT /= np.trapz(qOUT, ex)
##	## Normalise by size of region
###	qIN /= x[cuspind]-x[0]
###	qOUT /= x[-1]-x[cuspind]
##	## Plot
###	ax.plot(ex, qIN, "b-", label=labels[0])
###	ax.fill_between(ex,0,qIN,facecolor="blue",alpha=0.1)
###	ax.plot(ex, qOUT, "g-", label=labels[1])
###	ax.fill_between(ex,0,qOUT,facecolor="green",alpha=0.1)
#	## Lots of intermediate
#	colours = ["r","k","b","k","grey","orange","grey","k","b"]
#	linesty = ["-"]*6+["--"]*3
#	for i,idx in enumerate([0,cuspind/2,cuspind,3*cuspind/2,Sind,(Sind+Rind)/2,Rind,Rind+cuspind/2,Rind+cuspind]):
#		ax.plot(ex, H[idx], c=colours[i], ls=linesty[i], label="%.2f"%(x[idx]))
#	ax.set_ylim(0,1.5*H[Sind].max())
##	##
##	##---------------------------------------------------------------	
		
	## Accoutrements
	ax.yaxis.set_major_locator(MaxNLocator(7))
	ax.set_xlabel(r"$\eta$")
	ax.set_ylabel(r"$p(\eta)$")
	ax.grid()
	
	## Make legend if standalone
	if not nosave:
		ax.legend()
		
	##---------------------------------------------------------------
	## Plot inset
	if not nosave:
		if "_ML_" in histfile:
			## Plot potential as inset
			x = np.linspace(-R-1.5,+R+1.5,x.size)
			fx = force_mlin([x,0],R,S,T)[0]
			U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
			left, bottom, width, height = [0.18, 0.63, 0.25, 0.25]
			axin = fig.add_axes([left, bottom, width, height])
			axin.plot(x, U, "k-")
#			axin.axvspan(x[0],x[cuspind], color=lL[0].get_color(),alpha=0.2)
#			axin.axvspan(x[cuspind],x[-1], color=lR[0].get_color(),alpha=0.2)
			axin.set_xlim(-R-1.5, R+1.5)
#			axin.set_ylim(top=2*U[cuspind])
			axin.xaxis.set_major_locator(NullLocator())
			axin.yaxis.set_major_locator(NullLocator())
			axin.set_xlabel(r"$x$", fontsize = fs["fsa"]-5)
			axin.set_ylabel(r"$U$", fontsize = fs["fsa"]-5)
		
	##---------------------------------------------------------------

	# fig.suptitle(r"PDF of $\eta$, divided into regions. $\alpha="+str(a)+"$, $R="+str(R)+"$, $S="+str(S)+"$")

	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFeta"+os.path.basename(histfile)[4:-4]+".pdf"
		fig.savefig(plotfile)
		print me+"Figure saved to",plotfile
	
	return


##=================================================================================================

##=================================================================================================
if __name__=="__main__":
	main()
