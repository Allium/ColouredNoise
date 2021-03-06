me0 = "LE_CPressure"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, optparse, glob, time

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator

from LE_CSim import force_dlin, force_clin, force_mlin, force_nlin, force_dcon
from LE_Utils import filename_par, fs, set_mplrc

from schem_force import plot_U1D_Cartesian
from schem_force import UP_CL, UP_ML

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

## Plot defaults
set_mplrc(fs)

## ============================================================================

def main():
	"""
	Plot the marginalised densities Q(x), qx(etax) and  qy(etay).
	Adapted from LE_PDFre.py.
	"""
	me = me0+".main: "
	t0 = time.time()
	
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('--str',
		dest="srchstr", default="", type="str")
	parser.add_option('--logplot',
		dest="logplot", default=False, action="store_true")
	parser.add_option('--jpg',
		dest="savejpg", default=False, action="store_true")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('--noread',
		dest="noread", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	srchstr = opt.srchstr
	logplot = opt.logplot
	nosave = opt.nosave
	noread = opt.noread
	vb = opt.verbose
	
	if opt.savejpg: fs["saveext"]="jpg"
	
	## Plot file
	if os.path.isfile(args[0]):
		plot_pressure_file(args[0], nosave, vb)
	## Plot all files
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_CAR_*"+srchstr+"*.npy"))
		if vb: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_pressure_file(histfile, nosave, vb)
			plt.close()
	## Plot directory
	elif os.path.isdir(args[0]):
		plot_pressure_dir(args[0], srchstr, logplot, nosave, noread, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def plot_pressure_file(histfile, nosave, vb):
	"""
	Plot spatial PDF Q(x) and spatially-varying pressure P(x).
	"""
	me = me0+".plot_pressure_file: "
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert "_CAR_" in histfile, me+"Functional only for Cartesian geometry."
	Casimir = "_CL_" in histfile or "_ML_" in histfile or "_NL_" in histfile

	##-------------------------------------------------------------------------
	
	## Filename parameters
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	try: T = filename_par(histfile, "_T")
	except ValueError: T = -S
			
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
		
	## Wall indices
	Rind, Sind, Tind = np.abs(x-R).argmin(), np.abs(x-S).argmin(), np.abs(x-T).argmin()
	STind = (Sind+Tind)/2
	
	## Adjust indices for pressure calculation
	if "_DC_" in histfile:
		STind = 0
	elif "_DL_" in histfile:
		STind = 0
	elif "_NL_" in histfile:
		STind = Sind
		Sind = Rind
		Tind = x.size-Rind
		
	##-------------------------------------------------------------------------
	
	## Histogram
	H = np.load(histfile)
	## Spatial density
	Qx = H.sum(axis=2).sum(axis=1) / (H.sum()*(x[1]-x[0]))
	
	##-------------------------------------------------------------------------
	
	## Choose force
	if   "_DC_" in histfile:	fx = force_dcon([x,0],R,S)[0]
	elif "_DL_" in histfile:	fx = force_dlin([x,0],R[i],S[i])[0]
	elif "_CL_" in histfile:	fx = force_clin([x,0],R,S,T)[0]
	elif "_ML_" in histfile:	fx = force_mlin([x,0],R,S,T)[0]
	elif "_NL_" in histfile:	fx = force_nlin([x,0],R,S)[0]
	else: raise IOError, me+"Force not recognised."
		
	## Calculate integral pressure
	PR = -sp.integrate.cumtrapz(fx[Rind:]*Qx[Rind:], x[Rind:], initial=0.0)
	PS = -sp.integrate.cumtrapz(fx[STind:Sind+1]*Qx[STind:Sind+1], x[STind:Sind+1], initial=0.0); PS -= PS[-1]
	if Casimir:
		PT = -sp.integrate.cumtrapz(fx[Tind:STind+1]*Qx[Tind:STind+1], x[Tind:STind+1], initial=0.0)
	
	if x[0]<0:
		R2ind = x.size-Rind
		PR2 = -sp.integrate.cumtrapz(fx[:R2ind]*Qx[:R2ind], x[:R2ind], initial=0.0); PR2 -= PR2[-1]
			
	##-------------------------------------------------------------------------
	
	## Potential and WN
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
	Qx_WN = np.exp(-U) / np.trapz(np.exp(-U), x)
	
	## WN pressure
	PR_WN = -sp.integrate.cumtrapz(fx[Rind:]*Qx_WN[Rind:], x[Rind:], initial=0.0)
	PS_WN = -sp.integrate.cumtrapz(fx[STind:Sind+1]*Qx_WN[STind:Sind+1], x[STind:Sind+1], initial=0.0); PS_WN -= PS_WN[-1]
	if Casimir:
		PT_WN = -sp.integrate.cumtrapz(fx[Tind:STind+1]*Qx_WN[Tind:STind+1], x[Tind:STind+1], initial=0.0)
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	fig, axs = plt.subplots(2,1, sharex=True, figsize=fs["figsize"])
	
	if   "_DL_" in histfile:	legloc = "upper right"
	elif "_CL_" in histfile:	legloc = "upper right"
	elif "_ML_" in histfile:	legloc = "upper left"
	elif "_NL_" in histfile:	legloc = "lower left"
	else:						legloc = "best"
	
	## Plot PDF
	ax = axs[0]
	lQ = ax.plot(x, Qx, lw=2, label=r"CN")
	ax.plot(x, Qx_WN, lQ[0].get_color()+":", lw=2, label="WN")
	## Potential
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--", lw=2, label=r"$U(x)$")
	
	ax.set_xlim((x[0],x[-1]))	
	ax.set_ylim(bottom=0.0)	
	ax.set_ylabel(r"$Q(x)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc=legloc, fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	
	## Plot pressure
	ax = axs[1]
	lPR = ax.plot(x[Rind:], PR, lw=2, label=r"$P_R$")
	lPS = ax.plot(x[STind:Sind+1], PS, lw=2, label=r"$P_S$")
	if Casimir:
		lPT = ax.plot(x[Tind:STind+1], PT, lw=2, label=r"$P_T$")
	if x[0]<0:
		ax.plot(x[:R2ind], PR2, lPR[0].get_color()+"-", lw=2)
	## WN result
	ax.plot(x[Rind:], PR_WN, lPR[0].get_color()+":", lw=2)
	ax.plot(x[STind:Sind+1], PS_WN, lPS[0].get_color()+":", lw=2)
	if Casimir:
		ax.plot(x[Tind:STind+1], PT_WN, lPT[0].get_color()+":", lw=2)
	if x[0]<0:
		ax.plot(x[:R2ind], PR_WN[::-1], lPR[0].get_color()+":", lw=2)
	## Potential
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--", lw=2)#, label=r"$U(x)$")
	
	ax.set_xlim((x[0],x[-1]))	
	ax.set_ylim(bottom=0.0)	
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$P(x)$", fontsize=fs["fsa"])
	ax.grid()
	if Casimir:
		ax.legend(loc=legloc, fontsize=fs["fsl"]).get_frame().set_alpha(0.5)

	##-------------------------------------------------------------------------
	
	fig.tight_layout()
	fig.subplots_adjust(top=0.90)
	title = r"Spatial PDF and Pressure. $\alpha=%.1f, R=%.1f, S=%.1f, T=%.1f$"%(a,R,S,T) if T>=0.0\
			else r"Spatial PDF and Pressure. $\alpha=%.1f, R=%.1f, S=%.1f$"%(a,R,S)
	fig.suptitle(title, fontsize=fs["fst"])
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFP"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	return
	
##=============================================================================
def calc_pressure_dir(histdir, srchstr, noread, vb):
	"""
	Calculate the pressure for all files in directory matching string.
	The 
	"""
	me = me0+".calc_pressure_dir: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert "_CAR_" in histdir, me+"Functional only for Cartesian geometry."
	Casimir = "_DL_" not in histdir
	
	## File discovery
	filelist = np.sort(glob.glob(histdir+"/BHIS_CAR_*"+srchstr+"*.npy"))
	numfiles = len(filelist)
	assert numfiles>1, me+"Check input directory."
	if vb: print me+"found",numfiles,"files"

	##-------------------------------------------------------------------------
	
	A, R, S, T, PR, PS, PT, PU, PR_WN, PS_WN, PT_WN, PU_WN = np.zeros([12,numfiles])
	
	## Retrieve data
	for i, histfile in enumerate(filelist):
		
		ti = time.time()
		
		## Assuming R, S, T are same for all files
		A[i] = filename_par(histfile, "_a")
		R[i] = filename_par(histfile, "_R")
		S[i] = filename_par(histfile, "_S")
		try: 
			T[i] = filename_par(histfile, "_T")
		except ValueError:
			T[i] = -S[i]
			
		## Space
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		xbins = bins["xbins"]
		x = 0.5*(xbins[1:]+xbins[:-1])
		
		## Wall indices
		Rind, Sind, Tind = np.abs(x-R[i]).argmin(), np.abs(x-S[i]).argmin()+1, np.abs(x-T[i]).argmin()
		STind = 0 if T[i]<0.0 else (Sind+Tind)/2
	
		## Adjust indices for pressure calculation
		if "_DC_" in histfile:
			STind = 0
		elif "_DL_" in histfile:
			STind = 0
		elif "_NL_" in histfile:
			STind = Sind
			Sind = Rind
			Tind = x.size-Rind
		
		##-------------------------------------------------------------------------
		
		## Histogram
		H = np.load(histfile)
		## Spatial density
		Qx = H.sum(axis=2).sum(axis=1) / (H.sum()*(x[1]-x[0]))
		
		##-------------------------------------------------------------------------
		
		## Choose force
		if   "_DC_" in histfile:	fx = force_dcon([x,0],R[i],S[i])[0]
		elif "_DL_" in histfile:	fx = force_dlin([x,0],R[i],S[i])[0]
		elif "_CL_" in histfile:	fx = force_clin([x,0],R[i],S[i],T[i])[0]
		elif "_ML_" in histfile:	fx = force_mlin([x,0],R[i],S[i],T[i])[0]
		elif "_NL_" in histfile:	fx = force_nlin([x,0],R[i],S[i])[0]
		else: raise IOError, me+"Force not recognised."
		
		## Calculate integral pressure
		PR[i] = -sp.integrate.trapz(fx[Rind:]*Qx[Rind:], x[Rind:])
		PS[i] = +sp.integrate.trapz(fx[STind:Sind]*Qx[STind:Sind], x[STind:Sind])
		PT[i] = -sp.integrate.trapz(fx[Tind:STind]*Qx[Tind:STind], x[Tind:STind])
		if "_ML_" in histfile:
			mRind = x.size-Rind	## Index of wall at x=-R
			PU[i] = +sp.integrate.trapz(fx[:mRind]*Qx[:mRind], x[:mRind])
		
		if vb: print me+"a=%.1f:\tPressure calculation %.2g seconds"%(A[i],time.time()-ti)
		
		## Potential
		U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
		Qx_WN = np.exp(-U) / np.trapz(np.exp(-U), x)
		## WN pressure
		PR_WN[i] = -sp.integrate.trapz(fx[Rind:]*Qx_WN[Rind:], x[Rind:])
		PS_WN[i] = +sp.integrate.trapz(fx[STind:Sind]*Qx_WN[STind:Sind], x[STind:Sind])
		if Casimir:
			PT_WN[i] = -sp.integrate.trapz(fx[Tind:STind]*Qx_WN[Tind:STind], x[Tind:STind])
		if "_ML_" in histfile:
			PU_WN[i] = +sp.integrate.trapz(fx[:mRind]*Qx_WN[:mRind], x[:mRind])
		
	##-------------------------------------------------------------------------
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	R, S, T = R[srtidx], S[srtidx], T[srtidx]
	PR, PS, PT, PU = PR[srtidx], PS[srtidx], PT[srtidx], PU[srtidx]
	PR_WN, PS_WN, PT_WN, PU_WN = PR_WN[srtidx], PS_WN[srtidx], PT_WN[srtidx], PU_WN[srtidx]
	
	## Normalise
	PR /= PR_WN + (PR_WN==0)
	PS /= PS_WN + (PS_WN==0)
	if Casimir:
		PT /= PT_WN + (PT_WN==0)
	if "_ML_" in histdir:
		PU /= PU_WN + (PU_WN==0)
		
	##-------------------------------------------------------------------------
		
	## SAVING
	if not noread:
		pressfile = histdir+"/PRESS_"+srchstr+".npz"
		np.savez(pressfile, A=A, R=R, S=S, T=T, PR=PR, PS=PS, PT=PT, PU=PU,
								PR_WN=PR_WN, PS_WN=PS_WN, PT_WN=PT_WN, PU_WN=PU_WN)
		if vb:
			print me+"Calculations saved to",pressfile
			print me+"Calculation time %.1f seconds."%(time.time()-t0)

	return {"A":A,"R":R,"S":S,"T":T,"PR":PR,"PS":PS,"PT":PT,"PU":PU,
					"PR_WN":PR_WN,"PS_WN":PS_WN,"PT_WN":PT_WN,"PU_WN":PU_WN}
		

##=============================================================================
def plot_pressure_dir(histdir, srchstr, logplot, nosave, noread, vb):
	"""
	Plot the pressure for all files in directory matching string.
	"""
	me = me0+".plot_pressure_dir: "
	t0 = time.time()
		
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh
		
	try:
		assert noread == False
		pressdata = np.load(histdir+"/PRESS_"+srchstr+".npz")
		print me+"Pressure data file found:",histdir+"/PRESS_"+srchstr+".npz"
	except (IOError, AssertionError):
		print me+"No pressure data found. Calculating from histfiles."
		pressdata = calc_pressure_dir(histdir, srchstr, noread, vb)
		
	A = pressdata["A"]
	R = pressdata["R"]
	S = pressdata["S"]
	T = pressdata["T"]
	PR = pressdata["PR"]
	PS = pressdata["PS"]
	PT = pressdata["PT"]
	PU = pressdata["PU"]
	PR_WN = pressdata["PR_WN"]
	PS_WN = pressdata["PS_WN"]
	PT_WN = pressdata["PT_WN"]
	PU_WN = pressdata["PU_WN"]
	del pressdata
		
	Casimir = "_DL_" not in histdir and "_DC_" not in histdir

	##-------------------------------------------------------------------------
	## FIT if DL
	
	if 0 and not Casimir:
		## Fit log
		fitfunc = lambda x, m, c: m*x + c
		Au = np.unique(A) + int(logplot)
		for Si in np.unique(S):
			fitPR = sp.optimize.curve_fit(fitfunc, np.log(1+Au), np.log(PR[S==Si]), p0=[-0.5,+1.0])[0]
			fitPS = sp.optimize.curve_fit(fitfunc, np.log(1+Au), np.log(PS[S==Si]), p0=[-0.5,+1.0])[0]
			if vb:	print me+"Fit S=%.1f: PR=%.1f*(1+a)^(%.2f), PS=%.1f*(1+a)^(%.2f)"%\
						(Si,np.exp(fitPR[1]),fitPR[0],np.exp(fitPS[1]),fitPS[0])
			
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	t0 = time.time()
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	sty = ["-","--",":"]
	
	## Add a=0 point
	if 0.0 not in A:
		nlin = np.unique(S).size
		A = np.hstack([[0.0]*nlin,A])
		R = np.hstack([R[:nlin],R])
		S = np.hstack([S[:nlin],S])
		T = np.hstack([T[:nlin],T])
		PR = np.hstack([[1.0]*nlin,PR])
		PS = np.hstack([[1.0]*nlin,PS])
		PT = np.hstack([[1.0]*nlin,PT])
		PU = np.hstack([[1.0]*nlin,PU])
	
	Au = np.unique(A) + int(logplot)
	
	##-------------------------------------------------------------------------
	
	## Hold R & T fixed and vary S
	if np.unique(R).size==1:
		
		plotfile = histdir+"/PAS_R%.1f_S%.1f_T%.1f."%(R[0],S[0],T[0])+fs["saveext"] if T[0]>=0.0\
					else histdir+"/PAS_R%.1f_S%.1f."%(R[0],S[0])+fs["saveext"]
		title = r"Pressure as a function of $\alpha$ for $R=%.1f,T=%.1f$"%(R[0],T[0]) if T[0]>=0.0\
				else r"Pressure as a function of $\alpha$ for $R=%.2f$"%(R[0])
		
		## To plot a single S
		if np.unique(S).size==1:
			ax.plot(Au, PR, "gv--", label=r"$P_R$", zorder=2)
			ax.plot(Au, PS, "go-", label=r"$P_S$", zorder=2)
			ax.plot(Au, PT, "bo-", label=r"$P_T$", zorder=2)
			if "_ML_" in histdir:
				ax.plot(Au, PU, "bv--", label=r"$P_U$", zorder=2)
				ax.plot(Au, -(PR-PS+PT-PU), "ks:", label=r"Net", zorder=2)
			
			##---------------------------------
			## Casimir insets
			if "_CL_" in histdir:
				## Pressure key
				left, bottom, width, height = [0.25, 0.15, 0.63, 0.4]
				axin = fig.add_axes([left, bottom, width, height])
				UP_CL(axin,3.0,2.0,0.0)
				axin.patch.set_alpha(0.3)
				## Potential sketch
				left, bottom, width, height = [0.18, 0.75, 0.25, 0.13]
				axin = fig.add_axes([left, bottom, width, height])
				x = np.linspace(-R[0]-2,+R[0]+2,1000)
				fx = force_clin([x,0],R[0],S[0],T[0])[0]
				U = -sp.integrate.cumtrapz(fx,x,initial=0.0); U-=U.min()
				axin.plot(x, U, "k-", lw=2, zorder=1)
				axin.set_xlim(x[0],x[-1])
	#			axin.set_ylim(0,1.2*ax.get_ylim()[1])
				axin.set_xlabel(r"$x$", fontsize=fs["fsa"]-4)
				axin.set_ylabel(r"$U$", fontsize=fs["fsa"]-4)
				axin.xaxis.set_major_locator(NullLocator())
				axin.yaxis.set_major_locator(NullLocator())	
				axin.patch.set_alpha(0.3)
			##---------------------------------
			## Single wall insets
			elif "_ML_" in histdir:
				## Plot potential as inset
				Uleft, Ubottom, Uwidth, Uheight = [0.21, 0.18, 0.30, 0.21]
				Rschem, Sschem, Tschem = 4.0, 2.0, 0.0
				x = np.linspace(-Rschem-2.0,+Rschem+2.0,501)
				fx = force_mlin([x,0],Rschem,Sschem,Tschem)[0]
				U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
				cuspind = np.abs(x-0.5*(Sschem+Tschem)).argmin()
				axin = fig.add_axes([Uleft, Ubottom, Uwidth, Uheight])
				axin.plot(x, U, "k-")
				axin.axvspan(x[0],x[cuspind], color="b",alpha=0.2)
				axin.axvspan(x[cuspind],x[-1], color="g",alpha=0.2)
				axin.set_xlim(x[0], x[-1])
				axin.set_ylim(top=3*U[cuspind])
				axin.xaxis.set_major_locator(NullLocator())
				axin.set_yticks([1.0])
				axin.set_yticklabels(["1"])
				axin.grid()
				axin.set_xlabel(r"$x$", fontsize = fs["fsa"]-5)
				axin.set_ylabel(r"$U/T$", fontsize = fs["fsa"]-5)
				## Pressure key
#				left, bottom, width, height = [0.25, 0.15, 0.63, 0.4]
#				axin = fig.add_axes([left, bottom, width, height])
#				UP_ML(axin,R[0],S[0],T[0])
#				axin.patch.set_alpha(0.1)
			##---------------------------------
			
		## If S varies
		else:			
			for Si in np.unique(S)[::-1]:
				if Casimir:
					ax.plot(Au, PR[S==Si], "o"+sty[0], label=r"$S=%.1f$"%(Si))
					ax.plot(Au, PS[S==Si], "o"+sty[1], c=ax.lines[-1].get_color())
					ax.plot(Au, PT[S==Si], "o"+sty[2], c=ax.lines[-1].get_color())
				else:	## If DL, label should be bulk width
					### One line, average.
					P = 0.5*(PR+PS)
					ax.plot(Au, PR[S==Si], "o"+sty[0], label=r"$L=%.1f$"%(R[0]-Si))
#					ax.plot(Au, PR[S==Si], "o"+sty[0], label=r"$S=%.1f$"%(Si))
#					ax.plot(Au, PS[S==Si], "o"+sty[1], c=ax.lines[-1].get_color())
					
					## Inset of potential
					left, bottom, width, height = [0.19, 0.58, 0.35, 0.30]
					axin = fig.add_axes([left, bottom, width, height])
					if "_DL_" in histdir: plot_U1D_Cartesian(axin, "dlin", 2.0, 0.0, 0.0)
					elif "_DC_" in histdir: plot_U1D_Cartesian(axin, "dcon", 1.0, -1.0, 0.0)
##					axin.patch.set_alpha(0.5)
			
			## Prediction for zero and infinite bulk
			if "_DL_" in histdir:
				ax.plot(Au, (Au)**(-0.5), "--", c=ax.lines[0].get_color())
				ax.plot(Au, np.ones(Au.size), "y--", label=r"$L\to\infty$")
			if "_DC_" in histdir:
				ax.plot(Au, np.ones(Au.size), "y--", label=r"$L\to\infty$")
					
			
	##-------------------------------------------------------------------------
	
	## Plot appearance
			
	if logplot:
		ax.set_xscale("log"); ax.set_yscale("log")
		ax.set_xlim((ax.get_xlim()[0],A[-1]+1))
		xlabel = r"$1+\frac{k\tau}{\zeta}$"
		plotfile = plotfile[:-4]+"_loglog."+fs["saveext"]
	else:
		ax.set_xlim((0.0,A[-1]))
		ax.set_ylim(bottom=0.0,top=max(1.2*ax.get_ylim()[1],1.0))
		xlabel = r"$\alpha$"
#	ax.set_ylim(1e-1,1e1)
	
	ax.set_xlabel(xlabel, fontsize=fs["fsa"])
	ax.set_ylabel(r"$P/P^{\rm passive}$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="best", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
#	fig.suptitle(title, fontsize=fs["fst"])
	
	if not nosave:
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Plotting time %.1f seconds."%(time.time()-t0)
	
	return
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
