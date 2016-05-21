
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, glob, optparse
import warnings
from time import time

from LE_Utils import save_data, filename_pars
from LE_SBS import force_const, force_lin, force_dcon, force_dlin,\
					force_tan, force_dtan, force_nu, force_dnu


warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in log",
	RuntimeWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in power",
	RuntimeWarning)

## Global variables
from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

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
		-v	--verbose
		-s	--show
			--nosave	False
		-a	--plotall
	"""
	me = "LE_SPressure.main: "
	t0 = time()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-P','--plotpress',
		dest="plotP", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('-h','--help',
		dest="help", default=False, action="store_true")		
	opt, args = parser.parse_args()
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	plotP	= opt.plotP
	verbose = opt.verbose
	nosave	= opt.nosave
	plotall = opt.plotall
	
	if plotall and os.path.isdir(args[0]):
		showfig = False
		allfiles(args[0],plotP,verbose)
		
	elif os.path.isfile(args[0]):
		pressure_pdf_file(args[0],plotP,verbose)
	elif os.path.isdir(args[0]):
		pressure_dir(args[0],nosave,verbose)
	else:
		raise IOError, me+"You gave me rubbish. Abort."
	
	if verbose: print me+"execution time",round(time()-t0,2),"seconds"
	if showfig: plt.show()
	
	return
	
##=============================================================================
def pressure_pdf_file(histfile, plotpress, verbose):
	"""
	Make plot for a single file
	"""
	me = "LE_SPressure.pressure_pdf_file: "
	t0 = time()

	## Filename
	plotfile = os.path.dirname(histfile)+"/PDFP"+os.path.basename(histfile)[4:-4]+".png"
	
	## Get pars from filename
	pars = filename_pars(histfile)
	[a,ftype,R,S,lam,nu] = [pars[key] for key in ["a","ftype","R","S","lam","nu"]]
	assert (R is not None), me+"You are using the wrong program. R must be defined."
	if verbose: print me+"alpha =",a,"and R =",R
	
	if S is None: S = 0.0
	fpars = [R,S,lam,nu]
		
	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	rmax = rbins[-1]
	r = 0.5*(rbins[1:]+rbins[:-1])
	erbins = bins["erbins"]
	er = 0.5*(erbins[1:]+erbins[-1])
	rini = 0.5*(max(rbins[0],S)+R)	## Start point for computing pressures
	rinid = np.argmin(np.abs(r-rini))
	dr = r[1]-r[0]
	
	## Load histogram, convert to normalised pdf
	H = np.load(histfile).sum(axis=2)
	## Noise dimension irrelevant here
	H = np.trapz(H, x=er, axis=1)
	## rho is probability density. H is probability at r
	rho = H/(2*np.pi*r) / np.trapz(H, x=r, axis=0)

	## White noise result
	r_WN = np.linspace(dr,r[-1]+0.5*(r[1]-r[0]),r.size*5+1)
	rho_WN = pdf_WN(r_WN,fpars,ftype,verbose)
	
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
		plotfile = plotfile[:-4]+"_P.png"
	figtit += ftype+"; $\\alpha="+str(a)+"$, $R = "+str(R)+"$"
	if ftype[0]   == "d":	figtit += ", $S = "+str(S)+"$"
	if ftype[-3:] == "tan": figtit += ", $\\lambda="+str(lam)+"$"
	if ftype[-2:] == "nu":  figtit += ", $\\lambda="+str(lam)+"$, $\\nu="+str(nu)+"$"
		
	##---------------------------------------------------------------	
	## PDF PLOT
	
	## Wall
	plot_wall(ax, ftype, fpars, r)
	## PDF and WN PDF
	ax.plot(r,rho,   "b-", label="CN simulation")
	ax.plot(r_WN,rho_WN,"r-", label="WN theory")
	## Accoutrements
	ax.set_xlim(right=rmax)
	ax.set_ylim(bottom=0.0, top=min(20,round(max(rho.max(),rho_WN.max())+0.05,1)))
	if not plotpress: ax.set_xlabel("$r$", fontsize=fsa)
	ax.set_ylabel("$\\rho(r,\\phi)$", fontsize=fsa)
	ax.grid()
	# ax.legend(loc="upper right",fontsize=fsl)
	
	##---------------------------------------------------------------
	## PRESSURE
	
	if plotpress:
	
		## Calculate force array
		if ftype == "const":	force = force_const(r,r,R)
		elif ftype == "lin":	force = force_lin(r,r,R)
		elif ftype == "lico":	force = force_lico(r,r,R)
		elif ftype == "dcon":	force = force_dcon(r,r,R,S)
		elif ftype == "dlin":	force = force_dlin(r,r,R,S)
		elif ftype == "tan":	force, force_WN = force_tan(r,r,R,lam), force_tan(r_WN,r_WN,R,lam)
		elif ftype == "dtan":	force, force_WN = force_dtan(r,r,R,S,lam), force_dtan(r_WN,r_WN,R,S,lam)
		elif ftype == "nu":		force, force_WN = force_nu(r,r,R,lam,nu), force_nu(r_WN,r_WN,R,lam,nu)
		elif ftype == "dnu":	force, force_WN = force_dnu(r,r,R,S,lam,nu), force_dnu(r_WN,r_WN,R,S,lam,nu)
		
		## Pressure array
		p		= -np.array([np.trapz(force[:i]*rho[:i], x=r[:i]) for i in xrange(r.shape[0])])
		p_WN	= -np.array([np.trapz(force_WN[:i]*rho_WN[:i], x=r_WN[:i]) for i in xrange(r_WN.shape[0])])
					
		## Eliminate negative values
		if ftype[0] == "d":
			p		-= p.min()
			p_WN	-= p_WN.min()
		
		##-----------------------------------------------------------
		## PRESSURE PLOT
		ax = axs[1]
		## Wall
		plot_wall(ax, ftype, fpars, r)
		## Pressure and WN pressure
		ax.plot(r,p,"b-",label="CN simulation")
		ax.plot(r_WN,p_WN,"r-",label="WN theory")
		## Accoutrements
		ax.set_xlim(left=0.0,right=rmax)
		ax.set_ylim(bottom=0.0, top=round(max(p.max(),p_WN.max())+0.05,1))
		ax.set_xlabel("$r$", fontsize=fsa)
		ax.set_ylabel("$P(r)$", fontsize=fsa)
		ax.grid()
		ax.legend(loc="upper left",fontsize=fsl)
	
	##---------------------------------------------------------------
	
	## Tidy figure
	fig.suptitle(figtit,fontsize=fst)
	fig.tight_layout();	plt.subplots_adjust(top=0.9)	
		
	fig.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
	
	return
	
##=============================================================================
def allfiles(dirpath, plotP, verbose):
	for filepath in glob.glob(dirpath+"/BHIS_CIR_*a2.0*.npy"):
		pressure_pdf_file(filepath, plotP, verbose)
		plt.close()
	return

##=============================================================================
def pressure_dir(dirpath, nosave, verbose):
	"""
	Plot pressure at "infinity" against alpha for all files in directory.
	"""
	me = "LE_SPressure.pressure_dir: "
	t0 = time()
	
	## Directory parameters
	dirpars = filename_pars(dirpath)
	ftype, geo = dirpars["ftype"], dirpars["geo"]
	
	## File discovery
	histfiles = np.sort(glob.glob(dirpath+"/BHIS_CIR_*.npy"))
	numfiles = len(histfiles)
	if verbose: print me+"found",numfiles,"files"
		
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
		
	## Loop over files
	for i,histfile in enumerate(histfiles):
	
		## Get pars from filename
		pars = filename_pars(histfile)
		[A[i],R[i],S[i],L[i],N[i]] = [pars[key] for key in ["a","R","S","lam","nu"]]

		## Space (for axes)
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		rbins = bins["rbins"]
		rmax = rbins[-1]
		r = 0.5*(rbins[1:]+rbins[:-1])
		erbins = bins["erbins"]
		er = 0.5*(erbins[1:]+erbins[-1])
		## Start point for computing pressures
		bidx = np.argmin(np.abs(r-0.5*(max(rbins[0],S[i])+R[i])))
		dr = r[1]-r[0]
		
		## Load histogram, normalise
		H = np.load(histfile)
		H = np.trapz(H, x=er, axis=1)
		## Noise dimension irrelevant here; convert to *pdf*
		rho = H/(2*np.pi*r) / np.trapz(H, x=r, axis=0)
		
		rho_WN = pdf_WN(r,[R[i],S[i],L[i],N[i]],ftype)
				
		## Calculate force array
		if ftype == "const":	force = force_const(r,r,R[i])
		elif ftype == "lin":	force = force_lin(r,r,R[i])
		elif ftype == "lico":	force = force_lico(r,r,R[i],g)
		elif ftype == "dcon":	force = force_dcon(r,r,R[i],S[i])
		elif ftype == "dlin":	force = force_dlin(r,r,R[i],S[i])
		elif ftype == "tan":	force = force_tan(r,r,R[i],L[i])
		elif ftype == "dtan":	force = force_dtan(r,r,R[i],S[i],L[i])
		elif ftype == "nu":		force = force_nu(r,r,R[i],L[i],N[i])
		elif ftype == "dnu":	force = force_nu(r,r,R[i],S[i],L[i],N[i])
		
		## Pressure array
		P[i]    = -sp.integrate.simps((force*rho)[bidx:],    r[bidx:])
		P_WN[i] = -sp.integrate.simps((force*rho_WN)[bidx:], r[bidx:])
		if ftype[0] is "d":
			## Inner pressure
			Q[i]    = +sp.integrate.simps((force*rho)[:bidx],    r[:bidx])
			Q_WN[i] = +sp.integrate.simps((force*rho_WN)[:bidx], r[:bidx])
				
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
						
	## 3D pressure array for DNU force: [L,R,A]
	## Assume all N equal
	elif ftype == "dnu":
		PP = -np.ones([NN.size,LL.size,RR.size,AA.size])
		QQ = -np.ones([NN.size,LL.size,RR.size,AA.size])
		PP_WN = np.zeros(PP.shape)
		QQ_WN = np.zeros(QQ.shape)
		for j in range(LL.size):
			Lidx = (L==LL[j])
			for k in range(RR.size):
				Ridx = (R==RR[k])
				for l in range(AA.size):
					Aidx = (A==AA[l])
					Pidx = Nidx*Lidx*Ridx*Aidx
					try:
						PP[i,j,k,l] = P[Pidx]
						PP_WN[i,j,k,l] = P_WN[Pidx]
					except ValueError:
						## No value there
						pass
					
		## Mask zeros
		PP_WN = np.ma.array(PP_WN, mask = PP==-1)
		QQ_WN = np.ma.array(QQ_WN, mask = QQ==-1)
		PP = np.ma.array(PP, mask = PP==-1)
		QQ = np.ma.array(QQ, mask = QQ==-1)
		
	## ------------------------------------------------
	## PLOTS
	
	fig, ax = plt.subplots(1,1)
	
	## Default labels etc.
	PP /= PP_WN
	if  ftype[0] is "d": QQ /= QQ_WN
	title = "Pressure normalised by WN; ftype = "+ftype
	plotfile = dirpath+"/PAR.png"
	ylabel = "Pressure"
	if ftype == "lin" or ftype == "lico" or ftype == "dlin":
		#xlabel = "$\\alpha=k\\tau/\\zeta$"
		xlabel = "$\\alpha$"
	else:
		#xlabel = "$\\alpha=f_0^2\\tau/T\\zeta$"
		xlabel = "$\\alpha$"
	xlim = (AA[0],AA[-1])
	DPplot = False
	
	## ------------------------------------------------
	## Plot pressure
			
	## E2 prediction
	if ftype == "lin":
		plt.plot(AA,np.power(AA+1.0,-0.5),"b:",label="$(\\alpha+1)^{-1/2}$",lw=2)
	
	## Single circus; non-finite
	## Plot pressure against alpha for R or for S
	if (ftype[0] != "d" and ftype[-3:] != "tan" and ftype[-2:] != "nu"):
		for i in range(RR.size):
			ax.plot(AA,PP[i,:],  "o-", label="$R = "+str(RR[i])+"$")
	
	## Double circus; non-finite		
	elif (ftype[0] == "d" and ftype[-3:] != "tan" and ftype[-2:] != "nu"):
		## Holding R fixed
		if RR.size == 1:
			title = "Pressure normalised by WN, $R = "+str(RR[0])+"$; ftype = "+ftype
			for i in range(SS.size):
				ax.plot(AA,PP[i,0,:],  "o-", label="$S = "+str(SS[i])+"$") 
				ax.plot(AA,QQ[i,0,:], "o--", color=ax.lines[-1].get_color())
		## Constant interval
		elif np.unique(RR-SS).size == 1:
			DPplot = True
			PP *= PP_WN; QQ *= QQ_WN
			title = "Pressure difference, $P_R-P_S$; $R-S = "+str((RR-SS)[0])+"$; ftype = "+ftype
			##
			"""plotfile = dirpath+"/DPAS.png"
			for i in range(SS.size):		## To plot against alpha
				ax.plot(AA,np.diagonal(PP-QQ).T[i,:], "o-", label="$S = "+str(SS[i])+"$")
				ax.plot(AA,np.diagonal(PP_WN-QQ_WN).T[i,:], "--", color=ax.lines[-1].get_color())
			"""
			plotfile = dirpath+"/DPSA.png"
			xlabel = "$S\\;(=R-"+str((RR-SS)[0])+")$"; ylabel = "Pressure Difference"; xlim = (SS[0],SS[-1])
			for i in range(0,AA.size,2):	## To plot against S
				ax.plot(SS,np.diagonal(PP-QQ).T[:,i], "o-", label="$\\alpha = "+str(AA[i])+"$") 
				ax.plot(SS,np.diagonal(PP_WN-QQ_WN).T[:,i], "--", color=ax.lines[-1].get_color())
				ax.plot(SS,(RR-SS)[0]*np.sqrt(AA[i])/(SS*SS),":", color=ax.lines[-1].get_color(),linewidth=2)
	
	## Single circus; TAN
	elif ftype == "tan":
		plotfile = dirpath+"/PAL.png"
		title = "Pressure normalised by WN, $R = "+str(RR[0])+"$; ftype = "+ftype
		for i in range(LL.size):
			ax.plot(AA,PP[i,0,:],  "o-", label="$\\lambda = "+str(LL[i])+"$")
	
	## Double circus; DTAN
	elif (ftype == "dtan"):
		## Dodgy points			
		ax.axvspan(0.0,0.2,color="r",alpha=0.1)
		## P[L,R,S,A]
		plotbool = [0,0,0,1]
		## R,S fixed
		if plotbool[0]:
			if verbose:	print me+"Plotting P_R and P_S against alpha for different values of lambda. R and S fixed."
			Ridx = 1
			Sidx = np.where(SS==RR[Ridx]-1.0)[0][0]
			plotfile = dirpath+"/PAL.png"
			title = "Pressure normalised by WN, $R = "+str(RR[Ridx])+"$, $S = "+str(SS[Sidx])+"$; ftype = "+ftype
			for i in range(LL.size):
				ax.plot(AA,PP[i,Ridx,Sidx,:], "o-", label="$\\lambda = "+str(LL[i])+"$")
				ax.plot(AA,QQ[i,Ridx,Sidx,:], "o--", color=ax.lines[-1].get_color())
		## lam fixed; R-S fixed
		elif plotbool[1]:
			if verbose:	print me+"Plotting P_R and P_S against alpha for different values of R. lambda is fixed."
			Lidx = 0
			plotfile = dirpath+"/PAR.png"
			title = "Pressure normalised by WN, $\\lambda = "+str(LL[0])+"$; ftype = "+ftype
			for i in range(RR.size):
				Sidx = np.where(SS==RR[i]-1.0)[0][0]
				ax.plot(AA,PP[0,i,Sidx,:], "o-", label="$R,S = "+str(RR[i])+", "+str(SS[Sidx])+"$")
				ax.plot(AA,QQ[0,i,Sidx,:], "o--", color=ax.lines[-1].get_color())
		## Difference in pressure; R,S fixed
		elif plotbool[2]:
			DPplot = True
			if verbose:	print me+"Plotting P_R-P_S against alpha for different values of lambda. R-S fixed."
			PP *= PP_WN; QQ *= QQ_WN
			DP = PP-QQ
			## Find S indices corresponding to R-1.0
			Ridx = range(RR.size)
			Sidx = [np.where(SS==RR[Ridx[i]]-1.0)[0][0] for i in range(len(Ridx))]
			plotfile = dirpath+"/DPALDR.png"
			title = "Pressure difference, $P_R-P_S$; $R-S = "+str(RR[Ridx[0]]-SS[Sidx[0]])+"$; ftype = "+ftype
			for i in range(LL.size):
				ax.plot(AA,DP[i,Ridx[0],Sidx[0],:], "o-",
							label="$\\lambda = "+str(LL[i])+"$. $R,S = "+str(RR[Ridx[0]])+", "+str(SS[Sidx[0]])+"$")
				ax.plot(AA,DP[i,Ridx[1],Sidx[1],:], "o--", color=ax.lines[-1].get_color(),
							label="$\\lambda = "+str(LL[i])+"$. $R,S = "+str(RR[Ridx[1]])+", "+str(SS[Sidx[1]])+"$")	
				ax.plot(AA,DP[i,Ridx[2],Sidx[2],:], "o:", color=ax.lines[-1].get_color(),
							label="$\\lambda = "+str(LL[i])+"$. $R,S = "+str(RR[Ridx[2]])+", "+str(SS[Sidx[2]])+"$")
		elif plotbool[3]:
			print me+"ABORT"; exit()
			DPplot = True
			if verbose:	print me+"Plotting P_R-P_S against R for single value of lambda and several alpha. R-S fixed."
			DP = PP*PP_WN-QQ*QQ_WN
			## Fix lambda index
			Lidx = 1
			plotfile = dirpath+"/DPRA.png"
			title = "Pressure difference, $P_R-P_S$; $R-S = "+str(RR[Ridx[0]]-SS[Sidx[0]])+"$; ftype = "+ftype
			for i in range(0,AA.size,2):	## To plot against S
				ax.plot(SS,DP[Lidx,Ridx,Sidx,i], "o-", label="$\\alpha = "+str(AA[i])+"$") 
			xlabel = "$S\\;(=R-"+str((RR-SS)[0])+")$"; ylabel = "Pressure Difference"; xlim = (SS[0],SS[-1])
			
	## Single circus; NU
	elif ftype == "nu":
		plotfile = dirpath+"/PALN.png"
		title = "Pressure normalised by WN, $R = "+str(RR[0])+"$; ftype = "+ftype
		for i in range(LL.size):
			ax.plot(AA,PP[0,i,:],   "o-", label="$\\lambda, \\nu = "+str(LL[i])+", "+str(NN[0]) +"$")
			ax.plot(AA,PP[-1,i,:], "o--", color=ax.lines[-1].get_color(), label="$\\lambda, \\nu = "+str(LL[i])+", "+str(NN[-1])+"$")
	
	## Double circus; DNU
	elif (ftype == "dnu"):
		if np.unique(RR-SS).size == 1:
			PP *= PP_WN; QQ *= QQ_WN
			plotfile = dirpath+"/DPRAL.png"
			title = "Pressure difference, $P_R-P_S$; $R-S = "+str((RR-SS)[0])+"$; ftype = "+ftype
			xlabel = "$R\\;(=S+"+str((SS-RR)[0])+")$"; ylabel = "Pressure Difference"; xlim = (RR[0],RR[-1])
			for i in range(0,AA.size,2):
				ax.plot(RR,np.diagonal(PP-QQ).T[:,i], "o-", label="$\\alpha = "+str(AA[i])+"$") 
				ax.plot(RR,np.diagonal(PP_WN-QQ_WN).T[:,i], "--", color=ax.lines[-1].get_color())
	
		
	## ------------------------------------------------
	## Accoutrements
	
	#ax.set_xscale("log"); ax.set_yscale("log"); plotfile = plotfile[:-4]+"_loglog.png"

	ax.set_xlim(xlim)
	if not DPplot:	ax.set_ylim(bottom=0.0, top=max(ax.get_ylim()[1],1.0))
	
	ax.set_xlabel(xlabel,fontsize=fsa)
	ax.set_ylabel(ylabel,fontsize=fsa)
	
	ax.grid()
	ax.legend(loc="best",fontsize=fsl)
	plt.suptitle(title,fontsize=fst)
	
	#plt.tight_layout();	plt.subplots_adjust(top=0.9)
	if not nosave:
		fig.savefig(plotfile)
		if verbose: print me+"plot saved to",plotfile
		
	return

##=============================================================================
def pdf_WN(r,fpars,ftype,vb=False):
	"""
	Theoretical radial pdf of a white noise gas.
	"""
	me = "LE_SPressure.pdf_WN: "
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
		# rho0 = 1.0/(R-S+np.sqrt(np.pi/2)*(1.0+sp.special.erf(S/np.sqrt(2))))
		rho0 = 1.0/(2.0*np.pi) *  1.0/(np.exp(-0.5*S*S)+np.sqrt(np.pi/2)*S*sp.special.erf(S/np.sqrt(2))+0.5*R*R-0.5*S*S+np.sqrt(np.pi/2)*R)
		rho_WN = rho0 * np.hstack([np.exp(-0.5*(S-r[:Sind])**2),np.ones(Rind-Sind),np.exp(-0.5*(r[Rind:]-R)**2)])
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
		print me+"Functionality not available."
		rho_WN = np.zeros(r.size)
	return rho_WN
	
	
def plot_wall(ax, ftype, fpars, r):
	"""
	Plot the wall profile of type ftype on ax
	"""
	me = "LE_SPressure.plot_wall: "
	R, S, lam = fpars[:3]
	Ridx, Sidx, Lidx = np.abs(R-r).argmin(), np.abs(S-r).argmin(), np.abs(S-lam-r).argmin()
	if ftype is "const":
		ax.plot(r,np.hstack([np.zeros(Ridx),r[Ridx:]-R]),"k--",label="Potential")
	elif ftype is "lin":
		ax.plot(r,np.hstack([np.zeros(Ridx),0.5*np.power(r[Ridx:]-R,2.0)]),"k--",label="Potential")
	elif ftype is "dcon":
		"""NEEDS UPDATE"""
		ax.axvline(R,c="k",ls="--",label="Potential")
		ax.axvline(S,c="k",ls="--")
	elif ftype is "dlin":
		"""NEEDS UPDATE"""
		ax.plot(r,np.hstack([S-r[:Sidx],np.zeros(Ridx-Sidx),r[Ridx:]-R]),"k--",label="Potential")
	elif ftype is "tan":
		Ufn = lambda Dr: -1.0*np.log(np.cos(0.5*np.pi*(Dr)/lam))
		ax.plot(r,np.hstack([np.zeros(Ridx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	elif ftype is "dtan":
		Ufn = lambda Dr: -1.0*np.log(np.cos(0.5*np.pi*(Dr)/lam))
		ax.plot(r[Lidx:],np.hstack([Ufn(S-r[Lidx:Sidx]),np.zeros(Ridx-Sidx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	elif ftype is "nu":
		nu = fpars[3]
		Ufn = lambda Dr: -1.0*nu*(np.log(1.0-(Dr*Dr)/(lam*lam)))
		ax.plot(r,np.hstack([np.zeros(Ridx),Ufn(r[Ridx:]-R)]),"k--",label="Potential")
	return
	
	
##=============================================================================
if __name__=="__main__":
	main()
