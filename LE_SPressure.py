
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, glob, optparse
import warnings
from time import time

from LE_Utils import save_data, filename_pars
from LE_SBS import force_const, force_lin, force_dcon, force_dlin,\
					force_tan, force_dtan


warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in sign",
	RuntimeWarning)

## Global variables
from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

def main():
	"""
	NAME
		LE_SPressure.py
	
	PURPOSE
	
	EXECUTION
	
	ARGUMENTS
		histfile	path to density histogram
		directory 	path to directory containing histfiles
	
	OPTIONS
	
	FLAGS
		-v --verbose
		-s --show
		-a --plotall
	
	EXAMPLE
		
	NOTES
	
	BUGS / TODO
	
	HISTORY
		21/03/2016	Started
	"""
	me = "LE_SPressure.main: "
	t0 = time()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
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
	verbose = opt.verbose
	nosave	= opt.nosave
	plotall = opt.plotall
	
	if plotall and os.path.isdir(args[0]):
		showfig = False
		allfiles(args[0],verbose)
		
	if os.path.isfile(args[0]):
		pressure_pdf_file(args[0],verbose)
	elif os.path.isdir(args[0]):
		pressure_dir(args[0],nosave,verbose)
	else:
		raise IOError, me+"You gave me rubbish. Abort."
	
	if verbose: print me+"execution time",round(time()-t0,2),"seconds"
	if showfig: plt.show()
	
	return
	
##=============================================================================
def pressure_pdf_file(histfile, verbose):
	"""
	Make plot for a single file
	"""
	me = "LE_SPressure.pressure_pdf_file: "
	t0 = time()
	
	plotpress = True

	## Filename
	plotfile = os.path.dirname(histfile)+"/PDFP"+os.path.basename(histfile)[4:-4]+".png"
	
	## Get pars from filename
	pars = filename_pars(histfile)
	[a,ftype,R,S,lam] = [pars[key] for key in ["a","ftype","R","S","lam"]]
	assert (R is not None), me+"You are using the wrong program. R must be defined."
	if verbose: print me+"alpha =",a,"and R =",R
	
	if S is None: S = 0.0
	fpars = [R,S,lam]
		
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
	H = np.load(histfile)
	## Noise dimension irrelevant here
	H = np.trapz(H, x=er, axis=1)
	## rho is probability density. H is probability at r
	rho = H/r / np.trapz(H, x=r, axis=0)

	## White noise result
	rho_WN = pdf_WN(r,[R,S,lam],ftype)
	
	## Set up plot
	if not plotpress:
		## Only pdf plot
		figtit = "Density; "
		fig, ax = plt.subplots(1,1)
	elif plotpress:
		figtit = "Density and pressure; "
		fig, axs = plt.subplots(2,1,sharex=True)
		ax = axs[0]
	figtit += ftype+"; $\\alpha="+str(a)+"$, $R = "+str(R)+"$, $S = "+str(S)+"$"
	if ftype[-3:] == "tan": figtit += ", $\\lambda="+str(lam)+"$"
		
	## PDF PLOT
	## Wall
	plot_wall(ax, ftype, fpars, r)
	## PDF and WN PDF
	ax.plot(r,rho,   "b-", label="CN simulation")
	ax.plot(r,rho_WN,"r-", label="WN theory")
	## Accoutrements
	ax.set_xlim(right=rmax)
	ax.set_ylim(bottom=0.0, top=min(20,round(max(rho.max(),rho_WN.max())+0.05,1)))
	if not plotpress: ax.set_xlabel("$r$", fontsize=fsa)
	ax.set_ylabel("$\\rho(r,\\phi)$", fontsize=fsa)
	ax.grid()
	# ax.legend(loc="upper right",fontsize=fsl)
	
	if plotpress:
	
		## Calculate force array
		if ftype == "const":	force = force_const(r,r,R)
		elif ftype == "lin":	force = force_lin(r,r,R)
		elif ftype == "lico":	force = force_lico(r,r,R)
		elif ftype == "dcon":	force = force_dcon(r,r,R,S)
		elif ftype == "dlin":	force = force_dlin(r,r,R,S)
		elif ftype == "tan":	force = force_tan(r,r,R,lam)
		elif ftype == "dtan":	force = force_dtan(r,r,R,S,lam)
		
		## Pressure array
		p		= -np.array([np.trapz(force[:i]*rho[:i],   x=r[:i]) for i in xrange(r.shape[0])])
		p_WN	= -np.array([np.trapz(force[:i]*rho_WN[:i],x=r[:i]) for i in xrange(r.shape[0])])
		
		## Eliminate negative values -- is this useful?
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
		ax.plot(r,p_WN,"r-",label="WN theory")
		## Accoutrements
		ax.set_xlim(left=0.0,right=rmax)
		ax.set_ylim(bottom=0.0, top=round(max(p.max(),p_WN.max())+0.05,1))
		ax.set_xlabel("$r$", fontsize=fsa)
		ax.set_ylabel("$P(r)$", fontsize=fsa)
		ax.grid()
		ax.legend(loc="upper left",fontsize=fsl)
	
	## Tidy figure
	fig.suptitle(figtit,fontsize=fst)
	fig.tight_layout();	plt.subplots_adjust(top=0.9)	
		
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
	
	return
	
##=============================================================================
def allfiles(dirpath, verbose):
	for filepath in glob.glob(dirpath+"/BHIS_CIR_*.npy"):
		pressure_pdf_file(filepath, verbose)
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
	P = np.zeros(numfiles)	## Pressures on outer wall
	Q = np.zeros(numfiles)	## Pressures on inner wall
	P_WN = np.zeros(numfiles)
	Q_WN = np.zeros(numfiles)
		
	## Loop over files
	for i,histfile in enumerate(histfiles):
	
		## Get pars from filename
		pars = filename_pars(histfile)
		[A[i],R[i],S[i],lam] = [pars[key] for key in ["a","R","S","lam"]]

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
		H /= np.trapz(np.trapz(H, x=er, axis=1),x=r,axis=0)
		## Noise dimension irrelevant here; convert to *pdf*
		rho = np.trapz(H, x=er, axis=1)/r
		
		rho_WN = pdf_WN(r,[R[i],S[i],lam],ftype)

		## Calculate force array
		if ftype == "const":	force = force_const(r,r,R[i])
		elif ftype == "lin":	force = force_lin(r,r,R[i])
		elif ftype == "lico":	force = force_lico(r,r,R[i],g)
		elif ftype == "dcon":	force = force_dcon(r,r,R[i],S[i])
		elif ftype == "dlin":	force = force_dlin(r,r,R[i],S[i])
		elif ftype == "tan":	force = force_tan(r,r,R[i],lam)
		elif ftype == "dtan":	force = force_dtan(r,r,R[i],S[i],lam)
		
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
	
	## 2D pressure array: [R,A]
	if ftype[0] is not "d":
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
		## Mask zeros
		PP_WN = np.ma.array(PP_WN, mask = PP==-1)
		PP = np.ma.array(PP, mask = PP==-1)
	
	## 3D pressure array wall: [S,R,A]
	elif  ftype[0] is "d":
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
	plotfile = dirpath+"/PAR1.png"
	ylabel = "Pressure"
	if ftype == "lin" or ftype == "lico" or ftype == "dlin":
		#xlabel = "$\\alpha=k\\tau/\\zeta$"
		xlabel = "$\\alpha$"
	else:
		#xlabel = "$\\alpha=f_0^2\\tau/T\\zeta$"
		xlabel = "$\\alpha$"
	xlim = (AA[0],AA[-1])
	
	## ------------------------------------------------
	## Plot pressure
			
	## E2 prediction
	if ftype == "lin":
		plt.plot(AA,np.power(AA+1.0,-0.5),"b:",label="$(\\alpha+1)^{-1/2}$",lw=2)
	
	## Plot pressure against alpha for R or for S
	if ftype[0] is not "d":
		for i in range(RR.size):
			ax.plot(AA,PP[i,:],  "o-", label="$R = "+str(RR[i])+"$")
			
	elif ftype[0] is "d":
		## Holding R fixed
		if RR.size == 1:
			title = "Pressure normalised by WN, $R = "+str(RR[0])+"$; ftype = "+ftype
			for i in range(SS.size):
				ax.plot(AA,PP[i,0,:],  "o-", label="$S = "+str(SS[i])+"$") 
				ax.plot(AA,QQ[i,0,:], "o--", color=ax.lines[-1].get_color())
		## Constant interval
		elif np.unique(RR-SS).size == 1:
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
			#ax.set_xscale("log");ax.set_yscale("log");ax.set_ylim(bottom=1e-3)
			
	## ------------------------------------------------
	## Accoutrements
	
	ax.set_xlim(xlim)
	#if ftype=="const": ax.set_ylim(bottom=0.5,top=1.5)
	#elif ftype=="lin": ax.set_ylim(bottom=0.5)
	
	ax.set_xlabel(xlabel,fontsize=fsa)
	ax.set_ylabel(ylabel,fontsize=fsa)
	
	ax.grid()
	ax.legend(loc="best",fontsize=fsl)
	plt.suptitle(title,fontsize=fst)
	
	#plt.tight_layout();	plt.subplots_adjust(top=0.9)
	if not nosave:
		plt.savefig(plotfile)
		if verbose: print me+"plot saved to",plotfile
		
	return

##=============================================================================
def pdf_WN(r,fpars,ftype,vb=False):
	"""
	Theoretical radial pdf of a white noise gas.
	"""
	me = "LE_SPressure.pdf_WN: "
	R, S = fpars[:2]
	if ftype is "const":
		Rind = np.argmin(np.abs(r-R))
		# rho0 = 1.0/(R+1.0)
		rho0 = 1.0/(0.5*R*R+R+1.0)
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.exp(R-r[Rind:])])
	elif ftype is "lin":
		Rind = np.argmin(np.abs(r-R))
		# rho0 = 1.0/(R+np.sqrt(np.pi/2))
		rho0 = 1.0/(0.5*R*R+np.sqrt(np.pi/2)*R+1.0)
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.exp(-0.5*(r[Rind:]-R)**2)])
	elif ftype is "dcon":
		Rind, Sind = np.argmin(np.abs(r-R)), np.argmin(np.abs(r-S))
		# rho0 = 1.0/(R-S+2-np.exp(-S))
		rho0 = 1.0/(S+R+np.exp(-S)+0.5*R*R-0.5*S*S)
		rho_WN = rho0 * np.hstack([np.exp(r[:Sind]-S),np.ones(Rind-Sind),np.exp(R-r[Rind:])])
	elif ftype is "dlin":
		Rind, Sind = np.argmin(np.abs(r-R)), np.argmin(np.abs(r-S))
		# rho0 = 1.0/(R-S+np.sqrt(np.pi/2)*(1.0+sp.special.erf(S/np.sqrt(2))))
		rho0 = 1.0/(np.exp(-0.5*S*S)+np.sqrt(np.pi/2)*S*sp.special.erf(S/np.sqrt(2))+0.5*R*R-0.5*S*S+np.sqrt(np.pi/2)*R)
		rho_WN = rho0 * np.hstack([np.exp(-0.5*(S-r[:Sind])**2),np.ones(Rind-Sind),np.exp(-0.5*(r[Rind:]-R)**2)])
	elif ftype is "tan":
		lam = fpars[2]
		Rind = np.argmin(np.abs(r-R))
		# rho0 = 1.0/(R+2/np.sqrt(np.pi)*sp.special.gamma(0.5*(1.0+1.0))/sp.special.gamma(0.5*(1.0)))/(2*np.pi)
		rho0 = 1.0
		if vb:	print me+"Warning! Normalisation calculated numerically."
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.power(np.cos(0.5*np.pi*(r[Rind:]-R)/lam),lam)])
		rho_WN /= np.trapz(rho_WN*r, x=r, axis=0)
	elif ftype is "dtan":
		lam = fpars[2]
		Rind, Sind = np.argmin(np.abs(r-R)), np.argmin(np.abs(r-S))
		rho0 = 1.0
		if vb:	print me+"Warning! Normalisation calculated numerically."
		rho_WN = rho0 * np.hstack([np.power(np.cos(0.5*np.pi*(r[:Sind]-S)/lam),lam),\
					np.ones(Rind-Sind),np.power(np.cos(0.5*np.pi*(r[Rind:]-R)/lam),lam)])
		rho_WN /= np.trapz(rho_WN*r, x=r, axis=0)
	else:
		print me+"Functionality not written yet."
		rho_WN = np.zeros(r.size)
	return rho_WN
	
	
def plot_wall(ax, ftype, fpars, r):
	"""
	Plot the wall profile of type ftype on ax
	"""
	me = "LE_SPressure.plot_wall: "
	R, S, lam = fpars
	if ftype is "const":
		Ridx = np.argmin(np.abs(R-r))
		ax.plot(r,np.hstack([np.zeros(Ridx),r[Ridx:]-R]),"k--",label="Potential")
	elif ftype is "lin":
		Ridx = np.argmin(np.abs(R-r))
		ax.plot(r,np.hstack([np.zeros(Ridx),0.5*np.power(r[Ridx:]-R,2.0)]),"k--",label="Potential")
	elif ftype is "dcon":
		"""NEEDS UPDATE"""
		ax.axvline(R,c="k",ls="--",label="Potential")
		ax.axvline(S,c="k",ls="--")
	elif ftype is "dlin":
		"""NEEDS UPDATE"""
		Ridx, Sidx = np.argmin(np.abs(R-r)), np.argmin(np.abs(S-r))
		ax.plot(r,np.hstack([S-r[:Sidx],np.zeros(Ridx-Sidx),r[Ridx:]-R]),"k--",label="Potential")
	elif ftype is "tan":
		Ridx = np.argmin(np.abs(R-r))
		U_fn = lambda Dr: -1.0*np.log(np.cos(0.5*np.pi*(Dr)/lam))
		ax.plot(r,np.hstack([np.zeros(Ridx),U_fn(r[Ridx:]-R)]),"k--",label="Potential")
	elif ftype is "dtan":
		Ridx, Sidx = np.argmin(np.abs(R-r)), np.argmin(np.abs(S-r))
		U_fn = lambda Dr: -1.0*np.log(np.cos(0.5*np.pi*(Dr)/lam))
		ax.plot(r,np.hstack([U_fn(S-r[:Sidx]),np.zeros(Ridx-Sidx),U_fn(r[Ridx:]-R)]),"k--",label="Potential")
	return
	
	
##=============================================================================
if __name__=="__main__":
	main()
