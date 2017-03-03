me0 = "LE_SPressure"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, glob, optparse
import warnings
from time import time

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LE_Utils import save_data, filename_pars, fs, set_mplrc
from LE_SSim import force_const, force_lin, force_dcon, force_dlin,\
					force_tan, force_dtan, force_nu, force_dnu
from schem_force import plot_U3D_polar

## Ignore warnings
warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in log",
	RuntimeWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in power",
	RuntimeWarning)

## MPL defaults
set_mplrc(fs)

##=============================================================================
##=============================================================================
def main():
	"""
	NAME
		LE_SPressure.py
	
	EXECUTION
		python LE_SPressure.py [path] [flags]
	
	ARGUMENTS
		histfile	path to density histogram
		dirpath 	path to directory containing histfiles
		
	FLAGS
		-v	--vb	False
		-s	--show		False
			--nosave	False
		-a	--plotall	False
	"""
	me = me0+".main: "
	t0 = time()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option("-s","--show",
		dest="showfig", default=False, action="store_true")
	parser.add_option("-P","--plotpress",
		dest="plotP", default=False, action="store_true")
	parser.add_option("-v","--verbose",
		dest="verbose", default=False, action="store_true")
	parser.add_option("--logplot","--plotlog",
		dest="logplot", default=False, action="store_true")
	parser.add_option("--nosave",
		dest="nosave", default=False, action="store_true")
	parser.add_option('--noread',
		dest="noread", default=False, action="store_true")
	parser.add_option("-a","--plotall",
		dest="plotall", default=False, action="store_true")
	parser.add_option('--str',
		dest="searchstr", default="", type="str")
	parser.add_option("-h","--help",
		dest="help", default=False, action="store_true")		
	opt, args = parser.parse_args()
	if opt.help: print main.__doc__; return
	plotP	= opt.plotP
	showfig = opt.showfig
	plotall = opt.plotall
	srchstr = opt.searchstr
	logplot	= opt.logplot
	nosave	= opt.nosave
	noread	= opt.noread
	vb = opt.verbose
	
	if plotall and os.path.isdir(args[0]):
		showfig = False
		allfiles(args[0],plotP,srchstr,vb)
		
	elif os.path.isfile(args[0]):
		pressure_pdf_file(args[0],plotP,vb)
	elif os.path.isdir(args[0]):
		plot_pressure_dir(args[0], srchstr, logplot, nosave, noread, vb)
	else:
		raise IOError, me+"You gave me rubbish. Abort."
	
	if vb: print me+"execution time",round(time()-t0,2),"seconds"
	if showfig: plt.show()
	
	return
	
##=============================================================================
def pressure_pdf_file(histfile, plotpress, vb):
	"""
	Make plot for a single file
	"""
	me = me0+".pressure_pdf_file: "
	t0 = time()

	## Filename
	plotfile = os.path.dirname(histfile)+"/PDF"+os.path.basename(histfile)[4:-4]+"."+fs["saveext"]
	
	## Get pars from filename
	assert "_POL_" in histfile or "_CIR_" in histfile, me+"Check input file."
	pars = filename_pars(histfile)
	[a,ftype,R,S,lam,nu] = [pars[key] for key in ["a","ftype","R","S","lam","nu"]]
	assert (R is not None), me+"You are using the wrong program. R must be defined."
	if vb: print me+"alpha =",a,"and R =",R
	
	if S is None: S = 0.0
	fpars = [R,S,lam,nu]
		
	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	rmax = rbins[-1]
	r = 0.5*(rbins[1:]+rbins[:-1])
	erbins = bins["erbins"]
	er = 0.5*(erbins[1:]+erbins[:-1])
	rini = 0.5*(max(rbins[0],S)+R)	## Start point for computing pressures
	rinid = np.argmin(np.abs(r-rini))
	
	## Load histogram, convert to normalised pdf
	H = np.load(histfile)
	try:	H = H.sum(axis=2)
	except ValueError:	pass
	## Noise dimension irrelevant here
	H = np.trapz(H, x=er, axis=1)
	## rho is probability density. H is probability at r
	rho = H/(2*np.pi*r) / np.trapz(H, r, axis=0)

	## White noise result
	r_WN = np.linspace(r[0],r[-1]*(1+0.5/r.size),r.size*5+1)
	rho_WN = pdf_WN(r_WN,fpars,ftype,vb)
	
	## If we want density = 1.0 in bulk HACKY
	#rho /= rho[:np.argmin(np.abs(r-R))/2].mean()
	#rho_WN /= rho_WN[:np.argmin(np.abs(r-R))/2].mean()
	
	##---------------------------------------------------------------			
	## PLOT SET-UP
	
	if not plotpress:
		## Only pdf plot
		figtit = "Density; "
		fig, ax = plt.subplots(1,1)
	elif plotpress:
		figtit = "Density and pressure; "
		fig, axs = plt.subplots(2,1,sharex=True)
		ax = axs[0]
		plotfile = os.path.dirname(plotfile)+"/PDFP"+os.path.basename(plotfile)[3:]
	figtit += ftype+"; $\\alpha="+str(a)+"$, $R = "+str(R)+"$"
	if ftype[0]   == "d":	figtit += r", $S = "+str(S)+"$"
	if ftype[-3:] == "tan": figtit += r", $\lambda="+str(lam)+"$"
	if ftype[-2:] == "nu":  figtit += r", $\lambda="+str(lam)+"$, $\nu="+str(nu)+"$"
	xlim = [S-2*lam,R+2*lam] if (ftype[-3:]=="tan" or ftype[-2:]=="nu") else [S-4.0,R+4.0]
	xlim[0] = max(xlim[0],0.0)
		
		
	##---------------------------------------------------------------	
	
	## Calculate force array
	if ftype == "const":	force = force_const(r,r,R)
	elif ftype == "lin":	force = force_lin(r,r,R)
	elif ftype == "lico":	force = force_lico(r,r,R)
	elif ftype == "dcon":	force = force_dcon(r,r,R,S)
	elif ftype == "dlin":	force = force_dlin(r,r,R,S)
	elif ftype == "tan":	force = force_tan(r,r,R,lam)
	elif ftype == "dtan":	force = force_dtan(r,r,R,S,lam)
	elif ftype == "nu":		force = force_nu(r,r,R,lam,nu)
	elif ftype == "dnu":	force = force_dnu(r,r,R,S,lam,nu)
	else: raise ValueError, me+"ftype not recognised."
	U = -sp.integrate.cumtrapz(force, r, initial=0.0); U -= U.min()
	
	##---------------------------------------------------------------	
	## PDF PLOT
	
	## PDF and WN PDF
	sp.ndimage.gaussian_filter1d(rho,sigma=2.0,order=0,output=rho)
	ax.plot(r,rho,   "b-", label="OUP")
	ax.fill_between(r,0,rho,facecolor="b",alpha=0.1)
	ax.plot(r_WN,rho_WN,"r-", label="Passive")
	ax.fill_between(r_WN,0,rho_WN,facecolor="r",alpha=0.1)
	## Wall
	ax.plot(r, U/U.max()*ax.get_ylim()[1], "k--", label=r"$U(x)$")
	
	## Accoutrements
	ax.set_xlim(xlim)
	ax.set_ylim(bottom=0.0, top=min(20,1.2*max(rho.max(),rho_WN.max())))
	if not plotpress: ax.set_xlabel("$r$", fontsize=fs["fsa"])
#	ax.set_ylabel(r"$\rho(r,\phi)$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$n(r)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="upper right")
	
	
	##---------------------------------------------------------------
	## PRESSURE
	
	if plotpress:
	
		## CALCULATIONS
		p	= calc_pressure(r,rho,ftype,[R,S,lam,nu],spatial=True)
		p_WN = calc_pressure(r_WN,rho_WN,ftype,[R,S,lam,nu],spatial=True)
		## Eliminate negative values
		if ftype[0] == "d":
			p		-= p.min()
			p_WN	-= p_WN.min()
		# print [a,R],"\t",np.around([p[:20].mean(),p[-20:].mean()],6)
		
		##-----------------------------------------------------------
		## PRESSURE PLOT
		ax = axs[1]
		## Pressure and WN pressure
		ax.plot(r,p,"b-",label="CN simulation")
		ax.plot(r_WN,p_WN,"r-",label="WN theory")
		## Wall
		ax.plot(r, U/U.max()*ax.get_ylim()[1], "k--", label=r"$U(x)$")
#		plot_wall(ax, ftype, fpars, r)
		## Accoutrements
		# ax.set_ylim(bottom=0.0, top=round(max(p.max(),p_WN.max())+0.05,1))
		ax.set_ylim(bottom=0.0, top=min(20,float(1.2*max(p.max(),p_WN.max()))))
		ax.set_xlabel("$r$", fontsize=fs["fsa"])
		ax.set_ylabel("$P(r)$", fontsize=fs["fsa"])
		ax.grid()
	
		fig.tight_layout();	plt.subplots_adjust(top=0.9)	
	##---------------------------------------------------------------
	
#	fig.suptitle(figtit,fontsize=fs["fst"])
		
	fig.savefig(plotfile)
	if vb: print me+"plot saved to",plotfile
	
	return
	
##=============================================================================
def allfiles(dirpath, plotP, srchstr, vb):
	for filepath in np.sort(glob.glob(dirpath+"/BHIS_POL_*"+srchstr+"*.npy") + glob.glob(dirpath+"/BHIS_CIR_*"+srchstr+"*.npy")):
		pressure_pdf_file(filepath, plotP, vb)
		plt.close()
	return

##=============================================================================
def calc_pressure_dir(dirpath, srchstr, noread, vb):
	"""
	Plot pressure at "infinity" against alpha for all files in directory.
	"""
	me = me0+".calc_pressure_dir: "
	t0 = time()
	
	## Directory parameters
	dirpars = filename_pars(dirpath)
	ftype, geo = dirpars["ftype"], dirpars["geo"]
	
	## File discovery
	histfiles = np.sort(glob.glob(dirpath+"/BHIS_POL_*"+srchstr+"*.npy")+glob.glob(dirpath+"/BHIS_CIR_*"+srchstr+"*.npy"))
	numfiles = len(histfiles)
	if vb: print me+"found",numfiles,"files"

	## Initialise
	A = np.zeros(numfiles) 
	R = np.zeros(numfiles)	## Outer radii
	S = np.zeros(numfiles)	## Inner radii
	L = np.zeros(numfiles)	## Size of wall when finite
	N = np.zeros(numfiles)	## Wall strength parameter
	P = np.zeros(numfiles)	## Pressures on outer wall
	Q = np.zeros(numfiles)	## Pressures on inner wall
	P_WN = np.zeros(numfiles)
	Q_WN = np.zeros(numfiles)
		
	
	t0 = time()
	## Loop over files
	for i,histfile in enumerate(histfiles):
		
		## Get pars from filename
		pars = filename_pars(histfile)
		[A[i],R[i],S[i],L[i],N[i]] = [pars[key] for key in ["a","R","S","lam","nu"]]
		fpars = [R[i],S[i],L[i],N[i]]

		## Space (for axes)
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		rbins = bins["rbins"]
		rmax = rbins[-1]
		r = 0.5*(rbins[1:]+rbins[:-1])
		erbins = bins["erbins"]
		er = 0.5*(erbins[1:]+erbins[:-1])
		## Start point for computing pressures
		bidx = np.argmin(np.abs(r-0.5*(max(rbins[0],S[i])+R[i])))
		
		## Load histogram, normalise
		H = np.load(histfile)
		try: H = H.sum(axis=2)
		except ValueError: pass
		H = np.trapz(H, x=er, axis=1)
		## Noise dimension irrelevant here; convert to *pdf*
		rho = H/(2*np.pi*r) / np.trapz(H, x=r, axis=0)
		
		rho_WN = pdf_WN(r,fpars,ftype)
		
		## Pressure array
		P[i] 	= calc_pressure(r[bidx:],rho[bidx:],ftype,fpars)
		P_WN[i] = calc_pressure(r[bidx:],rho_WN[bidx:],ftype,fpars)
		if ftype[0] is "d":
			## Inner pressure
			Q[i] 	= -calc_pressure(r[:bidx],rho[:bidx],ftype,fpars)
			Q_WN[i] = -calc_pressure(r[:bidx],rho_WN[:bidx],ftype,fpars)
			
	if vb: print me+"Pressure calculations %.1f seconds."%(time()-t0)
		
	## ------------------------------------------------	
	## Create 2D pressure array and 1D a,R coordinate arrays

	## Ordered independent variable arrays
	AA = np.unique(A)
	RR = np.unique(R)
	SS = np.unique(S)
	LL = np.unique(L)
	NN = np.unique(N)
		
	## 2D pressure array: [R,A]
	if (ftype[0] != "d" and ftype[-3:] != "tan" and ftype[-2:] != "nu"):
		PP = -np.ones([RR.size,AA.size])
		PP_WN = np.zeros(PP.shape)
		QQ, QQ_WN = np.zeros(PP.shape), np.zeros(PP.shape)
		for i in range(RR.size):
			Ridx = (R==RR[i])
			for j in range(AA.size):
				Aidx = (A==AA[j])
				Pidx = Ridx*Aidx
				try:
					PP[i,j] = P[Pidx]
					PP_WN[i,j] = P_WN[Pidx]
				except ValueError:
					## No value there
					pass

	
	## 3D pressure array: [S,R,A]
	elif  (ftype[0] == "d" and ftype[-3:] != "tan" and ftype[-2:] != "nu"):
		PP = -np.ones([SS.size,RR.size,AA.size])
		QQ = -np.ones([SS.size,RR.size,AA.size])
		PP_WN = np.zeros(PP.shape)
		QQ_WN = np.zeros(QQ.shape)
		for i in range(SS.size):
			Sidx = (S==SS[i])
			for j in range(RR.size):
				Ridx = (R==RR[j])
				for k in range(AA.size):
					Aidx = (A==AA[k])
					Pidx = Sidx*Ridx*Aidx
					try:
						PP[i,j,k] = P[Pidx]
						QQ[i,j,k] = Q[Pidx]
						PP_WN[i,j,k] = P_WN[Pidx]
						QQ_WN[i,j,k] = Q_WN[Pidx]
					except ValueError:
						## No value there
						pass
						
	## 3D pressure array for TAN force: [L,R,A]
	elif ftype == "tan":
		PP = -np.ones([LL.size,RR.size,AA.size])
		QQ = -np.ones([LL.size,RR.size,AA.size])
		PP_WN = np.zeros(PP.shape)
		QQ_WN = np.zeros(QQ.shape)
		for i in range(LL.size):
			Lidx = (L==LL[i])
			for j in range(RR.size):
				Ridx = (R==RR[j])
				for k in range(AA.size):
					Aidx = (A==AA[k])
					Pidx = Lidx*Ridx*Aidx
					try:
						PP[i,j,k] = P[Pidx]
						QQ[i,j,k] = Q[Pidx]
						PP_WN[i,j,k] = P_WN[Pidx]
						QQ_WN[i,j,k] = Q_WN[Pidx]
					except ValueError:
						## No value there
						pass
	
	## 4D pressure array for TAN force: [L,R,S,A]
	elif ftype == "dtan":
		PP = -np.ones([LL.size,RR.size,SS.size,AA.size])
		QQ = -np.ones([LL.size,RR.size,SS.size,AA.size])
		PP_WN = np.zeros(PP.shape)
		QQ_WN = np.zeros(QQ.shape)
		for i in range(LL.size):
			Lidx = (L==LL[i])
			for j in range(RR.size):
				Ridx = (R==RR[j])
				for k in range(SS.size):
					Sidx = (S==SS[k])
					for l in range(AA.size):
						Aidx = (A==AA[l])
						Pidx = Lidx*Ridx*Sidx*Aidx
						try:
							PP[i,j,k,l] = P[Pidx]
							QQ[i,j,k,l] = Q[Pidx]
							PP_WN[i,j,k,l] = P_WN[Pidx]
							QQ_WN[i,j,k,l] = Q_WN[Pidx]
						except ValueError:
							## No value there
							pass
	
						
	## 3D pressure array for NU force: [N,L,A]
	## Assume all R equal
	elif ftype == "nu":
		PP = -np.ones([NN.size,LL.size,AA.size])
		QQ = -np.ones([NN.size,LL.size,AA.size])
		PP_WN = np.zeros(PP.shape)
		QQ_WN = np.zeros(QQ.shape)
		for i in range(NN.size):
			Nidx = (N==NN[i])
			for j in range(LL.size):
				Lidx = (L==LL[j])
				for k in range(AA.size):
					Aidx = (A==AA[k])
					Pidx = Nidx*Lidx*Aidx
					try:
						PP[i,j,k] = P[Pidx]
						PP_WN[i,j,k] = P_WN[Pidx]
					except ValueError:
						## No value there
						pass
						
	## 4D pressure array for DNU force: [N,L,R,A]
	## Assume all N equal
	elif ftype == "dnu":
		PP = -np.ones([NN.size,LL.size,RR.size,AA.size])
		QQ = -np.ones([NN.size,LL.size,RR.size,AA.size])
		PP_WN = np.zeros(PP.shape)
		QQ_WN = np.zeros(QQ.shape)
		for i in range(NN.size):
			Nidx = (N==NN[i])
			for j in range(LL.size):
				Lidx = (L==LL[j])
				for k in range(RR.size):
					Ridx = (R==RR[k])
					for l in range(AA.size):
						Aidx = (A==AA[l])
						Pidx = Nidx*Lidx*Ridx*Aidx
						try:
							PP[i,j,k,l] = P[Pidx]
							QQ[i,j,k,l] = Q[Pidx]
							PP_WN[i,j,k,l] = P_WN[Pidx]
							QQ_WN[i,j,k,l] = Q_WN[Pidx]
						except ValueError:
							## No value there
							pass
					
	# ## Mask zeros
	# mask = (PP==0.0)+(PP==-1.0)
	# PP_WN = np.ma.array(PP_WN, mask=mask)
	# QQ_WN = np.ma.array(QQ_WN, mask=mask)
	# PP = np.ma.array(PP, mask=mask)
	# QQ = np.ma.array(QQ, mask=mask)
	
	## SAVING
	if not noread:
		pressfile = dirpath+"/PRESS.npz"
		np.savez(pressfile, PP=PP, QQ=QQ, PP_WN=PP_WN, QQ_WN=QQ_WN, AA=AA, RR=RR, SS=SS, LL=LL, NN=NN)
		if vb:	print me+"Calculations saved to",pressfile
	
	return {"PP":PP, "QQ":QQ, "PP_WN":PP_WN, "QQ_WN":QQ_WN, "AA":AA, "RR":RR, "SS":SS, "LL":LL, "NN":NN}
		

##=============================================================================
def plot_pressure_dir(dirpath, srchstr, logplot, nosave, noread, vb):
	"""
	Plot some slice of pressure array.
	"""
	me = me0+".plot_pressure_dir: "
	
	try:
		assert noread == False
		pressdata = np.load(dirpath+"/PRESS.npz")
		print me+"Pressure data file found:",dirpath+"/PRESS.npz"
	except (IOError, AssertionError):
		print me+"No pressure data found. Calculating from histfiles."
		pressdata = calc_pressure_dir(dirpath, srchstr, noread, vb)
	ftype = filename_pars(dirpath)["ftype"]
	
	PP = pressdata["PP"]
	QQ = pressdata["QQ"]
	PP_WN = pressdata["PP_WN"]
	QQ_WN = pressdata["QQ_WN"]
	AA = pressdata["AA"]
	RR = pressdata["RR"]
	SS = pressdata["SS"]
	LL = pressdata["LL"]
	NN = pressdata["NN"]
	del pressdata

	## ------------------------------------------------
	
	## Mask zeros
	mask = (PP==-1.0)
	PP_WN = np.ma.array(PP_WN, mask=mask)
	QQ_WN = np.ma.array(QQ_WN, mask=mask)
	PP = np.ma.array(PP, mask=mask)
	QQ = np.ma.array(QQ, mask=mask)
		
	## ------------------------------------------------
	## PLOTS
	
	t0 = time()
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	## Default labels etc.
	PP /= PP_WN + 1*(PP_WN==0.0)
	if  ftype[0] is "d": QQ /= QQ_WN + 1*(QQ_WN==0.0)
	title = "Pressure normalised by WN; ftype = "+ftype
	plotfile = dirpath+"/PAR"
	ylabel = "Pressure (normalised)"
	xlabel = "$\\alpha$"
	xlim = [AA[0],AA[-1]]
	DPplot = False
	
	## ------------------------------------------------
	## Plot pressure
			
	## E2 prediction
#	if ftype == "lin":
#		plt.plot(AA,np.power(AA+1.0,-0.5),"b:",label=r"$(\alpha+1)^{-1/2}$",lw=2)
	
	## Disc; non-finite
	## Plot pressure against alpha for R or for S
#	if ((ftype[0]!="d" and SS.sum()!=0.0) or (ftype[0]=="d" and SS.sum()==0.0)  and ftype[-3:]!="tan" and ftype[-2:]!="nu"):
	if SS.sum()==0.0:
		## In case run with DL but S=0 for all files, will need to restructure array
		if SS.all()==0.0:
			PP=PP[0]
		## Shunt on the a=0 values
		if 0.0 not in AA:
			AA = np.hstack([0.0,AA])
			PP = np.vstack([np.ones(RR.size),PP.T]).T
			xlim = [AA[0],AA[-1]]
		
		## PLOTTING	
		## Plot against ALPHA
		if 0:
			for i in range(RR.size):
				ax.plot(AA,PP[i,:],  "o-", label=r"$R = "+str(RR[i])+"$")
				
		## Plot against R
		else:
			plotfile = dirpath+"PRA"
			for i in range(AA.size):
				ax.plot(RR,PP[:,i],  "o-", label=r"$\alpha = %.1f$"%(AA[i]))
			xlim = [RR[0],RR[-1]]
			ax.xaxis.set_major_locator(MaxNLocator(3))
			xlabel = r"$R$"
	
	## ------------------------------------------------
	## 
	
	## Annulus; finite; [S,R,A]
	elif (ftype[0] == "d" and ftype[-3:] != "tan" and ftype[-2:] != "nu"):
		QQ, QQ_WN = np.nan_to_num(QQ), np.nan_to_num(QQ_WN)
		
		## Holding R fixed
		if RR.size == 1:
			if 1:
				## Plot normalised Pout and Pin individually against ALPHA
				title = "Pressure normalised by WN, $R = "+str(RR[0])+"$; ftype = "+ftype
				plotfile = dirpath+"/PQAS"
				xlabel = r"$1+\alpha$" if logplot else r"$\alpha$"
				xlim = (int(logplot), int(logplot)+AA[-1])
				## Shunt on the a=0 values
				if 0.0 not in AA:
					AA = np.hstack([0.0,AA])
					PP = np.dstack([np.ones([SS.size,RR.size]),PP])
					QQ = np.dstack([np.ones([SS.size,RR.size]),QQ])
				for i in range(SS.size):
					ax.plot(int(logplot)+AA,PP[i,0,:],  "o-", label="$S = "+str(SS[i])+"$") 
					ax.plot(int(logplot)+AA,QQ[i,0,:], "v--", color=ax.lines[-1].get_color())
			elif 0:
				## Plot normalised Pout and Pin individually against S, for multiple ALPHA
				title = "Pressures $P_R,P_S$ (normalised); $R = "+str(RR[0])+"$; ftype = "+ftype
				plotfile = dirpath+"/PQSA"
				xlabel = r"$S$"
				xlim = [SS[0],SS[-1]]
				## Shunt on the a=0 values
				if 0.0 not in AA:
					AA = np.hstack([0.0,AA])
					PP = np.dstack([np.ones([SS.size,RR.size]),PP])
					QQ = np.dstack([np.ones([SS.size,RR.size]),QQ])
				## Avoid plotting QQ for S=0 point
				idx = 1 if 0.0 in SS else 0
				for i in range(0,AA.size,1):
					ax.plot(SS,PP[:,0,i], "o-", label=r"$\alpha = "+str(AA[i])+"$") 
					ax.plot(SS[idx:],QQ[idx:,0,i], "v--", color=ax.lines[-1].get_color())
			elif 1:
				## Plot difference Pout-Pin against ALPHA, for multiple S
				DPplot = True
				PP *= PP_WN; QQ *= QQ_WN
				title = r"Pressure difference, $(P_R-P_S)/P_R^{\rm wn}$; $R = "+str(RR[0])+"$; ftype = "+ftype
				ylabel = "Pressure Difference (normalised)"
				xlabel = r"$\alpha$"
				xlim = [0.0, AA[-1]]
				plotfile = dirpath+"/DPAS"
				## Messy shunting a=0 onto the plot because PP,QQ too complicated to modify directly
				if 0.0 not in AA and not logplot:
					DP_WN = ((PP_WN-QQ_WN)/(PP_WN))[:,0,:]	## Data for the a=0 point
					for i in range(SS.size):
						ax.plot(np.hstack([0.0,AA]),np.hstack([DP_WN[i,0],((PP-QQ)/(PP_WN))[i,0,:]]),
									"o-", label="$S = "+str(SS[i])+"$")
				else:
					for i in range(SS.size):
						ax.plot(AA,((PP-QQ)/(PP_WN))[i,0,:], "o-", label="$S = "+str(SS[i])+"$")
			else:
				## Plot difference Pout-Pin against S
				DPplot = True
				PP *= PP_WN; QQ *= QQ_WN
				title = r"Pressure difference,  $(P_R-P_S)/P_R^{\rm wn}$; $R = "+str(RR[0])+"$; ftype = "+ftype
				plotfile = dirpath+"/DPSA"
				xlim = [SS[0],SS[-1]]
				xlabel = r"$S$"
				ylabel = "Pressure Difference (normalised)"
				for i in range(0,AA.size,1):
					ax.plot(SS,(PP-QQ)[:,0,i]/(PP_WN)[:,0,i], "o-", label=r"$\alpha = "+str(AA[i])+"$")
					
		## Constant interval
		elif np.unique(RR-SS).size == 1:
			if 0:
				## Plot Pout and Pin individually against R
				title = "Pressures $P_R,P_S$ (normalised); $R-S = "+str((RR-SS)[0])+"$; ftype = "+ftype
				plotfile = dirpath+"/PQRA"
				xlabel = r"$R\;(=S+%.1f)$"%((RR-SS)[0]) if (RR-SS)[0]>0.0 else r"$R$"
				xlim = [RR[0],RR[-1]]
				for i in range(0,AA.size,1):	## To plot against R
					ax.plot(RR,np.diagonal(PP).T[:,i], "o-", label=r"$\alpha = "+str(AA[i])+"$") 
					ax.plot(RR,np.diagonal(QQ).T[:,i], "v--", color=ax.lines[-1].get_color())
			elif 0:
				## Plot difference Pout-Pin against ALPHA, for multiple S
				DPplot = True
				PP *= PP_WN; QQ *= QQ_WN
				if AA[0]==0.0 and logplot:
					AA = AA[1:]; PP = PP[:,:,1:]; QQ = QQ[:,:,1:]; PP_WN = PP_WN[:,:,1:]
				title = r"Pressure difference, $(P_R-P_S)/P_R^{\rm wn}$; $R-S = "+str((RR-SS)[0])+"$; ftype = "+ftype
				ylabel = "Pressure Difference (normalised)"
				legloc = "upper right"#(0.25,0.54)
				plotfile = dirpath+"/DPAS"
#				## Advance colour cycle for consistent colour between diffrent R-S plots
#				[ax.plot([],[]) for i in range(np.sum([Ri not in RR for Ri in [0.0,1.0,2.0,5.0]])-1)]
				## Messy shunting a=0 onto the plot because PP,QQ too complicated to modify directly
				if 0.0 not in AA and not logplot:
					if vb: print me+"Adding alpha=0 points."
					DP_WN = np.diagonal((PP_WN-QQ_WN)/(PP_WN)).T	## To plot the a=0 point
					for i in range(SS.size):
						ax.plot(np.hstack([0.0,AA]),np.hstack([DP_WN[i,0],np.diagonal((PP-QQ)/(PP_WN)).T[i,:]]),
									"o-", label="$R = "+str(RR[i])+"$")
					xlim[0]=0.0
				else:
					for i in range(SS.size):
						ax.plot(AA,np.diagonal((PP-QQ)/(PP_WN)).T[i,:],	"o-", label="$R = "+str(RR[i])+"$")
				
				## POTENTIAL INSET
				left, bottom, width, height = [0.53, 0.39, 0.33, 0.30] if np.unique(RR-SS) == 0 else [0.52, 0.43, 0.33, 0.31]
				axin = fig.add_axes([left, bottom, width, height], projection="3d")
				Rschem, Sschem = (1.7,1.7) if np.unique(RR-SS) == 0 else (5,1.7)
				plot_U3D_polar(axin, Rschem, Sschem)
				axin.patch.set_alpha(0.3)
					
			elif 0:
				## Plot difference Pout-Pin against R
				DPplot = True
				PP *= PP_WN; QQ *= QQ_WN
				if AA[0]==0.0:
					AA = AA[1:]; PP = PP[:,:,1:]; QQ = QQ[:,:,1:]; PP_WN = PP_WN[:,:,1:]
				title = r"Pressure difference, $(P_R-P_S)/P_R^{\rm wn}$; $R-S = "+str((RR-SS)[0])+"$; ftype = "+ftype
				ylabel = "Pressure Difference (normalised)"
				plotfile = dirpath+"/DPRA"
				xlabel = r"$R\;(=S+%.1f)$"%((RR-SS)[0]) if (RR-SS)[0]>0.0 else r"$R$"
				xlim = [RR[0],RR[-1]]
				for i in range(0,AA.size,1):	## To plot against R
					ax.plot(RR,np.diagonal((PP-QQ)/(PP_WN)).T[:,i], "o-", label=r"$\alpha = "+str(AA[i])+"$")
				if logplot:
					RRR = np.linspace(1,RR[-1],10)
					ax.plot(RRR,2/(RRR),"k:",lw=3,label=r"$2R^{-1}$")
#					ax.set_color_cycle(None)
#					for i in range(0,AA.size,1):
#						ax.plot(RR,2/(RR)*(AA[i]/(AA[i]+1))*(1+0.5*np.unique(RR-SS)/RR*np.sqrt(AA[i]/(AA[i]+1))),":",lw=3)
				ax.set_ylim(1e-3,1e1)
				xlim = [1.0,RR[-1]]
				
				## POTENTIAL INSET
				left, bottom, width, height = [0.51, 0.58, 0.35, 0.31]	## For upper right
#				left, bottom, width, height = [0.13, 0.14, 0.35, 0.31]	## For lower left
				axin = fig.add_axes([left, bottom, width, height], projection="3d")
				Rschem, Sschem = (1.7,1.7) if np.unique(RR-SS) == 0 else (5,1.7)
				plot_U3D_polar(axin, Rschem, Sschem)
				axin.patch.set_alpha(0.3)
				
			
			else:
				## SHURA REQUEST Plot difference (Pout-Pin)/(PE2*sqrt(a)) against R
				DPplot = True
				PP *= PP_WN; QQ *= QQ_WN
				ylabel = r"Pressure Difference (normalised)"
				plotfile = dirpath+"/DPRA_AG"
				xlabel = r"$R\;(=S+%.1f)$"%((RR-SS)[0]) if (RR-SS)[0]>0.0 else r"$R$"
				xlim = [RR[0],RR[-1]]
				PP_E2 = 1/np.sqrt(1+AA)[:,np.newaxis]/(2*np.pi*RR)	## /R for 2D
				for i in range(0,AA.size,1):	## To plot against R
					## Normalise by PEN
#					ax.plot(RR,np.diagonal(PP-QQ).T[:,i]/np.diagonal(PP_WN).T[:,i]/np.sqrt(AA[i]), "o-", label=r"$\alpha = "+str(AA[i])+"$")
					## Normalise by PE2
					ax.plot(RR,np.diagonal(PP-QQ).T[:,i]/PP_E2[i]/np.sqrt(AA[i]), "o-", label=r"$\alpha = "+str(AA[i])+"$")
					## Normalise by Pout
#					ax.plot(RR,np.diagonal(1-QQ/PP).T[:,i], "o-", label=r"$\alpha = "+str(AA[i])+"$")
					## Individual
#					ax.plot(RR,np.diagonal(PP/PP_WN).T[:,i],"o-", label=r"$\alpha = "+str(AA[i])+"$")
#					ax.plot(RR,np.diagonal(QQ/QQ_WN).T[:,i],"o--", color=ax.lines[-1].get_color())
#					ax.plot(RR,PP_E2[i,:]/np.diagonal(PP_WN).T[:,i],"o:", color=ax.lines[-1].get_color())
					##
				if logplot:
					RRR = np.linspace(1,RR[-1],10)
					ax.plot(RRR,1/(RRR),"k:",lw=3,label=r"$R^{-1}$")
#				ax.set_ylim(1e-5,1e0)
				xlim = [1.0,RR[-1]]
				fig.subplots_adjust(left=0.15)
				
				## POTENTIAL INSET
				left, bottom, width, height = [0.51, 0.60, 0.35, 0.31]	## For upper right
#				left, bottom, width, height = [0.13, 0.14, 0.35, 0.31]	## For lower left
				axin = fig.add_axes([left, bottom, width, height], projection="3d")
				Rschem, Sschem = (1.7,1.7) if np.unique(RR-SS) == 0 else (5,1.7)
				plot_U3D_polar(axin, Rschem, Sschem)
				axin.patch.set_alpha(0.3)
				axin.patch.set_facecolor("None")
			
	## ------------------------------------------------
	
	## Single circus; TAN
	elif ftype == "tan":
		plotfile = dirpath+"/PAL"
		title = "Pressure normalised by WN, $R = "+str(RR[0])+"$; ftype = "+ftype
		for i in range(LL.size):
			ax.plot(AA,PP[i,0,:],  "o-", label="$\\lambda = "+str(LL[i])+"$")
	
	## Double circus; DTAN
	elif (ftype == "dtan"):
		## Dodgy points			
		ax.axvspan(0.0,0.2,color="r",alpha=0.1)
		## P[L,R,S,A]
		plotbool = [0,0,1,1]
		## R,S fixed
		if plotbool[0]:
			if vb:	print me+"Plotting P_R and P_S against alpha for different values of lambda. R and S fixed."
			Ridx = 1
			Sidx = np.where(SS==RR[Ridx]-1.0)[0][0]
			plotfile = dirpath+"/PAL"
			title = "Pressure normalised by WN, $R = "+str(RR[Ridx])+"$, $S = "+str(SS[Sidx])+"$; ftype = "+ftype
			for i in range(LL.size):
				ax.plot(AA[1:],PP[i,Ridx,Sidx,1:], "o-", label="$\\lambda = "+str(LL[i])+"$")
				ax.plot(AA[1:],QQ[i,Ridx,Sidx,1:], "o--", color=ax.lines[-1].get_color())
		## lam fixed; R-S fixed
		elif plotbool[1]:
			if vb:	print me+"Plotting P_R and P_S against alpha for different values of R. lambda is fixed."
			Lidx = 0
			plotfile = dirpath+"/PAR"
			title = "Pressure normalised by WN, $\\lambda = "+str(LL[0])+"$; ftype = "+ftype
			for i in range(RR.size):
				Sidx = np.where(SS==RR[i]-1.0)[0][0]
				ax.plot(AA[1:],PP[0,i,Sidx,1:], "o-", label="$R,S = "+str(RR[i])+", "+str(SS[Sidx])+"$")
				ax.plot(AA[1:],QQ[0,i,Sidx,1:], "o--", color=ax.lines[-1].get_color())
		## Difference in pressure; R,S fixed
		elif plotbool[2]:
			DPplot = True
			if vb:	print me+"Plotting P_R-P_S against alpha for different values of lambda. R-S fixed."
			PP *= PP_WN; QQ *= QQ_WN
			DP = PP-QQ
			## Find S indices corresponding to R-1.0
			Ridx = range(RR.size)
			Sidx = [np.where(SS==RR[Ridx[i]]-1.0)[0][0] for i in range(len(Ridx))]
			plotfile = dirpath+"/DPALDR"
			title = "Pressure difference, $P_R-P_S$; $R-S = "+str(RR[Ridx[0]]-SS[Sidx[0]])+"$; ftype = "+ftype
			for i in range(LL.size):
				ax.plot(AA[1:],DP[i,Ridx[0],Sidx[0],1:], "o-",
							label="$\\lambda = "+str(LL[i])+"$. $R,S = "+str(RR[Ridx[0]])+", "+str(SS[Sidx[0]])+"$")
				ax.plot(AA[1:],DP[i,Ridx[1],Sidx[1],1:], "o--", color=ax.lines[-1].get_color(),
							label="$\\lambda = "+str(LL[i])+"$. $R,S = "+str(RR[Ridx[1]])+", "+str(SS[Sidx[1]])+"$")	
				ax.plot(AA[1:],DP[i,Ridx[2],Sidx[2],1:], "o:", color=ax.lines[-1].get_color(),
							label="$\\lambda = "+str(LL[i])+"$. $R,S = "+str(RR[Ridx[2]])+", "+str(SS[Sidx[2]])+"$")
		elif plotbool[3]:
			print me+"ABORT"; exit()
			DPplot = True
			if vb:	print me+"Plotting P_R-P_S against R for single value of lambda and several alpha. R-S fixed."
			DP = PP*PP_WN-QQ*QQ_WN
			## Fix lambda index
			Lidx = 1
			plotfile = dirpath+"/DPRA"
			title = "Pressure difference, $P_R-P_S$; $R-S = "+str(RR[Ridx[0]]-SS[Sidx[0]])+"$; ftype = "+ftype
			for i in range(0,AA.size,2):	## To plot against S
				ax.plot(SS,DP[Lidx,Ridx,Sidx,i], "o-", label="$\\alpha = "+str(AA[i])+"$") 
			xlabel = "$S\\;(=R-"+str((RR-SS)[0])+")$"; ylabel = "Pressure Difference"; xlim = (SS[0],SS[-1])
			
	## Disc; NU
	elif ftype == "nu":
		## All R equal; plot against a; large and small l
		"""
		plotfile = dirpath+"/PALN"
		title = "Pressure normalised by WN, $R = "+str(RR[0])+"$; ftype = "+ftype
		for i in range(LL.size):
			ax.plot(AA,PP[0,i,:],   "o-", label="$\\lambda, \\nu = "+str(LL[i])+", "+str(NN[0]) +"$")
			ax.plot(AA,PP[-1,i,:], "o--", color=ax.lines[-1].get_color(), label="$\\lambda, \\nu = "+str(LL[i])+", "+str(NN[-1])+"$")
		"""
		## All R equal; plot against nu; assume same L; [N,L,A]
		plotfile = dirpath+"/PNA"
		title = "Pressure normalised by WN, $R = "+str(RR[0])+"$; ftype = "+ftype+"; $\\lambda = "+str(LL[0])+"$"
		fitfunc = lambda NN, M, b: M*np.power(NN, b)
		for i in range(AA.size):
			print curve_fit(fitfunc, NN[1:], np.nan_to_num(PP[1:,0,i]))[0]
			ax.plot(NN,PP[:,0,i], "o-", label="$\\alpha = "+str(AA[i])+"$")
		xlabel = "$\\nu$"; xlim = (NN[0],NN[-1])
	
	## Annulus DNU [N,L,R,A]
	elif (ftype == "dnu"):
		## Pressure difference as function of R. Assume nu and lam equal.
		if (np.unique(RR-SS).size == 1 and PP.shape[0]==1 and PP.shape[1]==1):
			PP = PP[0,0]; QQ = QQ[0,0];	PP_WN = PP_WN[0,0]; QQ_WN = QQ_WN[0,0]
			if 0:
				## Plot individually
				title = "Pressures $P_R,P_S$ (normalised); $R-S = "+str((RR-SS)[0])+"$; ftype = "+ftype
				plotfile = dirpath+"/PQRA"
				xlabel = "$R\\;(=S+"+str((RR-SS)[0])+")$"; xlim = (RR[0],RR[-1])
				for i in range(0,AA.size,1):	## To plot against R
					ax.plot(RR[:],PP[:,i], "o-", label="$\\alpha = "+str(AA[i])+"$") 
					ax.plot(RR[1:],QQ[1:,i], "v--", color=ax.lines[-1].get_color())
				# if logplot:
					# ax.plot(RR,0.01/(RR),"k:",lw=3)
			else:
				## Plot difference
				DPplot = True
				PP *= PP_WN; QQ *= QQ_WN
				plotfile = dirpath+"/DPRA"
				title = "Pressure difference, $P_R-P_S$; $R-S = "+str((RR-SS)[0])+"$; ftype = "+ftype
				xlabel = "$R\\;(=S+"+str((RR-SS)[0])+")$"; ylabel = "Pressure Difference"
				xlim = (RR[0],RR[-1])
				for i in range(0,AA.size,1):
					ax.plot(RR,np.abs(PP-QQ)[:,i], "o-", label="$\\alpha = "+str(AA[i])+"$") 
					# ax.plot(RR,np.abs(PP_WN-QQ_WN)[:,i], "--", color=ax.lines[-1].get_color())
				if logplot:
					ax.plot(RR,0.1/(RR),"k:",lw=3)
					ax.plot(RR,0.1/(RR*RR),"k:",lw=3,label="$R^{-1}, R^{-2}$")
		else:
			raise IOError, me+"Check functionality for this plot exists."
		
	## ------------------------------------------------
	## Accoutrements
	
	if logplot:
		ax.set_xscale("log"); ax.set_yscale("log")
		ax.set_xlim(left=(xlim[0] if xlim[0]!=0.0 else ax.get_xlim()[0]), right=xlim[1])
		plotfile = plotfile+"_loglog"
	elif not (logplot or DPplot):
		ax.set_xlim(xlim)
		ax.set_ylim(bottom=0.0, top=max(ax.get_ylim()[1],1.0))
	else:
		ax.set_xlim(xlim)
	
	## AD HOC
	if not logplot:
		ax.set_ylim(bottom=0.0)
		ax.set_ylim(top=1.4)
#	ax.set_ylim(1e-3,1e1)
	
	ax.set_xlabel(xlabel,fontsize=fs["fsa"])
	ax.set_ylabel(ylabel,fontsize=fs["fsa"])
	
	ax.grid()
	try:
		ax.legend(loc=legloc,fontsize=fs["fsl"]-2, ncol=2).get_frame().set_alpha(0.5)
	except UnboundLocalError:
		ax.legend(loc="best",fontsize=fs["fsl"], ncol=2).get_frame().set_alpha(0.5)
#	fig.suptitle(title,fontsize=fs["fst"])
	
	#plt.tight_layout();	plt.subplots_adjust(top=0.9)
	if not nosave:
		fig.savefig(plotfile+"."+fs["saveext"])
		if vb: print me+"plot saved to",plotfile+"."+fs["saveext"]
	
	if vb: print me+"Plotting %.2f seconds."%(time()-t0)
		
	return

##=============================================================================

def calc_pressure(r,rho,ftype,fpars,spatial=False):
	"""
	Calculate pressure given density a a function of coordinate.
	"""
	me = me0+".calc_pressure: "
	R, S, lam, nu = fpars
	
	## Calculate force array
	if ftype == "const":	force = force_const(r,r,R)
	elif ftype == "lin":	force = force_lin(r,r,R)
	elif ftype == "lico":	force = force_lico(r,r,R)
	elif ftype == "dcon":	force = force_dcon(r,r,R,S)
	elif ftype == "dlin":	force = force_dlin(r,r,R,S)
	elif ftype == "tan":	force = force_tan(r,r,R,lam)
	elif ftype == "dtan":	force = force_dtan(r,r,R,S,lam)
	elif ftype == "nu":		force = force_nu(r,r,R,lam,nu)
	elif ftype == "dnu":	force = force_dnu(r,r,R,S,lam,nu)
	else: raise ValueError, me+"ftype not recognised."
	
	## Indices of wall (i.e. "valid" parts of distribution)
	if (ftype[-2:]=="nu" or ftype[:3]=="tan"):
		wibidx, wobidx = np.abs(S-lam-r).argmin(), np.abs(R+lam-r).argmin()
		force[:wibidx+1] = 0.0; force[wobidx:] = 0.0
	
	## Pressure
	if spatial == True:
#		P = -np.array([np.trapz(force[:i]*rho[:i], r[:i]) for i in range(1,r.size+1)])
		P = -sp.integrate.cumtrapz(force*rho, r, initial=0.0)
		# P = -np.array([sp.integrate.simps(force[:i]*rho[:i], r[:i]) for i in range(1,r.size+1)])
	else:
#		P = -np.trapz(force*rho, r)
		P = -sp.integrate.trapz(force*rho, r)
		# P = -sp.integrate.simps(force*rho, r)
	
	return P

##=============================================================================
def pdf_WN(r,fpars,ftype,vb=False):
	"""
	Theoretical radial pdf of a white noise gas.
	"""
	me = me0+".pdf_WN: "
	R, S = fpars[:2]
	Rind, Sind = np.argmin(np.abs(r-R)), np.argmin(np.abs(r-S))
	if ftype is "const":
		# rho0 = 1.0/(R+1.0)
		rho0 = 1.0/(2.0*np.pi) * 1.0/(0.5*R*R+R+1.0)
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.exp(R-r[Rind:])])
	elif ftype is "lin":
		# rho0 = 1.0/(R+np.sqrt(np.pi/2))
		rho0 = 1.0/(2.0*np.pi) * 1.0/(0.5*R*R+np.sqrt(np.pi/2)*R+1.0)
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.exp(-0.5*(r[Rind:]-R)**2)])
	elif ftype is "dcon":
		# rho0 = 1.0/(R-S+2-np.exp(-S))
		rho0 = 1.0/(2.0*np.pi) * 1.0/(S+R+np.exp(-S)+0.5*R*R-0.5*S*S)
		rho_WN = rho0 * np.hstack([np.exp(r[:Sind]-S),np.ones(Rind-Sind),np.exp(R-r[Rind:])])
	elif ftype is "dlin":
		rho0 = 1.0/(2.0*np.pi) *  1.0/(np.exp(-0.5*S*S)+np.sqrt(np.pi/2)*S*sp.special.erf(S/np.sqrt(2))+0.5*R*R-0.5*S*S+np.sqrt(np.pi/2)*R)
		rho_WN = rho0 * np.hstack([np.exp(-0.5*(S-r[:Sind])**2),np.ones(Rind-Sind),np.exp(-0.5*(r[Rind:]-R)**2)])
		# nu=10.0
		# rho_WN = np.hstack([np.exp(-0.5*nu*(S-r[:Sind])**2),np.ones(Rind-Sind),np.exp(-0.5*nu*(r[Rind:]-R)**2)])
		# rho_WN /= 2*np.pi*np.trapz(rho_WN*r, x=r, axis=0)
	elif ftype is "tan":
		lam = fpars[2]
		# rho0 = 1.0/(R+2/np.sqrt(np.pi)*sp.special.gamma(0.5*(1.0+1.0))/sp.special.gamma(0.5*(1.0)))/(2*np.pi)
		rho0 = 1.0
		if vb:	print me+"Normalisation calculated numerically."
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.nan_to_num(np.power(np.cos(0.5*np.pi*(r[Rind:]-R)/lam),lam))])
		rho_WN /= 2*np.pi*np.trapz(rho_WN*r, x=r, axis=0)
	elif ftype is "dtan":
		lam = fpars[2]
		Lind = np.abs(r-(S-lam)).argmin()
		rho0 = 1.0
		if vb:	print me+"Normalisation calculated numerically."
		rho_WN = rho0 * np.hstack([np.zeros(Lind),
					np.nan_to_num(np.power(np.cos(0.5*np.pi*(r[Lind:Sind]-S)/lam),lam)),\
					np.ones(Rind-Sind),
					np.nan_to_num(np.power(np.cos(0.5*np.pi*(r[Rind:]-R)/lam),lam))])
		rho_WN /= 2*np.pi*np.trapz(rho_WN*r, x=r, axis=0)
	elif ftype is "nu":
		lam, nu = fpars[2:4]
		rho0 = 1.0/(np.pi) * 1.0/(R*R+lam*lam/(lam*nu+1.0)+\
						+R*lam*np.sqrt(np.pi)*sp.special.gamma(nu*lam+1.0)/sp.special.gamma(nu*lam+1.5))
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.power(1.0-((r[Rind:]-R)/lam)**2.0,lam*nu)])
	else:
		lam, nu = fpars[2:4]
		LSind = np.abs(r-(S-lam)).argmin()
		LRind = np.abs(r-(R+lam)).argmin()
		rho0 = 1.0
		if vb:	print me+"Normalisation calculated numerically."
		rho_WN = rho0 * np.hstack(\
					[np.zeros(LSind),\
					np.power(1.0-((S-r[LSind:Sind])/lam)**2.0,lam*nu),\
					np.ones(Rind-Sind),\
					np.power(1.0-((r[Rind:LRind]-R)/lam)**2.0,lam*nu),\
					np.zeros(r.size-LRind)])
		rho_WN = np.nan_to_num(rho_WN)
		rho_WN /= 2*np.pi*np.trapz(rho_WN*r, x=r, axis=0)
	return rho_WN
	
	
def plot_wall(ax, ftype, fpars, r):
	"""
	Plot the wall profile of type ftype on ax
	"""
	me = me0+".plot_wall: "
	R, S, lam, nu = fpars
	Ridx, Sidx, Lidx = np.abs(R-r).argmin(), np.abs(S-r).argmin(), np.abs(S-lam-r).argmin()
	## Plot potentials
	if ftype is "const":
		ax.plot(r,np.hstack([np.zeros(Ridx),r[Ridx:]-R]),"k--",label="Potential")
	elif ftype is "dcon":
		ax.plot(r,np.hstack([S-r[:Sidx],np.zeros(Ridx-Sidx),r[Ridx:]-R]),"k--",label="Potential")
	elif ftype is "lin":
		Ufn = lambda Dr: 0.5*np.power(Dr,2.0)
		ax.plot(r,np.hstack([np.zeros(Ridx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	elif ftype is "dlin":
		Ufn = lambda Dr: 0.5*np.power(Dr,2.0)
		ax.plot(r,np.hstack([Ufn(S-r[:Sidx]),np.zeros(Ridx-Sidx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	## Infinite potentials
	elif ftype is "tan":
		Ufn = lambda Dr: -1.0*np.log(np.cos(0.5*np.pi*(Dr)/lam))
		ax.plot(r,np.hstack([np.zeros(Ridx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	elif ftype is "dtan":
		Ufn = lambda Dr: -1.0*np.log(np.cos(0.5*np.pi*(Dr)/lam))
		ax.plot(r[Lidx:],np.hstack([Ufn(S-r[Lidx:Sidx]),np.zeros(Ridx-Sidx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	elif ftype is "nu":
		Ufn = lambda Dr: -1.0*nu*(np.log(1.0-(Dr*Dr)/(lam*lam)))
		ax.plot(r,np.hstack([np.zeros(Ridx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	elif ftype is "dnu":
		Ufn = lambda Dr: -1.0*nu*(np.log(1.0-(Dr*Dr)/(lam*lam)))
		ax.plot(r,np.hstack([Ufn(S-r[:Sidx]),np.zeros(Ridx-Sidx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	return
	
	
##=============================================================================
if __name__=="__main__":
	main()
