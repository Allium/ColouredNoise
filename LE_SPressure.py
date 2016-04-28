
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, glob, optparse
import warnings
from time import time

from LE_Utils import save_data, filename_pars
from LE_Pressure import plot_wall
from LE_SBS import force_const, force_lin, force_lico, force_dcon, force_dlin


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
	[a,ftype,R,S] = [pars[key] for key in ["a","ftype","R","S"]]
	assert (R is not None), me+"You are using the wrong program. R must be defined."
	if verbose: print me+"alpha =",a,"and R =",R
	
	if S is None: S = 0.0
	fpars = [R,S]
		
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
	H = np.trapz(H/er, x=er, axis=1)
	## rho is probability density. H is probability at r
	# rho = Hr_norm(H/r,r,R)
	rho = H/r / np.trapz(H, x=r, axis=0)

	## White noise result
	rho_WN = pdf_WN(r,[R,S],ftype)
	
	## Set up plot
	if not plotpress:
		## Only pdf plot
		figtit = "Density for "+ftype+" wall; $\\alpha="+str(a)+"$, $R = "+str(R)+"$ and $S = "+str(S)+"$"
		fig,ax = plt.subplots(1,1)
	elif plotpress:
		figtit = "Density and pressure for "+ftype+" wall; $\\alpha="+str(a)+"$, $R = "+str(R)+"$ and $S = "+str(S)+"$"
		fig,axs = plt.subplots(2,1,sharex=True)
		ax = axs[0]
		
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
	ax.legend(loc="upper right",fontsize=fsl)
	
	if plotpress:
	
		## Calculate force array
		if ftype == "const":	force = force_const(r,r,r*r,R,R*R)
		elif ftype == "lin":	force = force_lin(r,r,r*r,R,R*R)
		elif ftype == "lico":	force = force_lico(r,r,r*r,R,R*R,g)
		elif ftype == "dcon":	force = force_dcon(r,r,r*r,R,R*R,S,S*S)
		elif ftype == "dlin":	force = force_dlin(r,r,r*r,R,R*R,S,S*S)
		
		## Pressure array
		p 	 = -np.array([np.trapz(force[:i]*rho[:i],   x=r[:i]) for i in xrange(r.shape[0])])
		p_WN = -np.array([np.trapz(force[:i]*rho_WN[:i],x=r[:i]) for i in xrange(r.shape[0])])
		
		## Eliminate negative values -- is this useful?
		if ftype == "dcon" or ftype == "dlin":
			p	 -= p.min()
			p_WN -= p_WN.min()
		
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
	
	## Tidy figure
	fig.suptitle(figtit,fontsize=fst)
	fig.tight_layout()
	plt.subplots_adjust(top=0.9)	
		
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
		[A[i],R[i],S[i]] = [pars[key] for key in ["a","R","S"]]

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
		
		rho_WN = pdf_WN(r,[R[i],S[i]],ftype)

		## Calculate force array
		if ftype == "const":	force = force_const(r,r,r*r,R[i],R[i]*R[i])
		elif ftype == "lin":	force = force_lin(r,r,r*r,R[i],R[i]*R[i])
		elif ftype == "lico":	force = force_lico(r,r,r*r,R[i],R[i]*R[i],g)
		elif ftype == "dcon":	force = force_dcon(r,r,r*r,R[i],R[i]*R[i],S[i],S[i]*S[i])
		elif ftype == "dlin":	force = force_dlin(r,r,r*r,R[i],R[i]*R[i],S[i],S[i]*S[i])
		
		## Pressure array
		P[i]    = -sp.integrate.simps((force*rho)[bidx:],    r[bidx:])
		P_WN[i] = -sp.integrate.simps((force*rho_WN)[bidx:], r[bidx:])
		if ftype == "dcon" or ftype == "dlin":
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
	if ftype == "const" or ftype == "lin" or ftype == "linco":
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
	elif  ftype == "dcon" or ftype == "dlin":
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
	if  ftype == "dcon" or ftype == "dlin": QQ /= QQ_WN
	title = "Pressure normalised by WN; ftype = "+ftype
	plotfile = dirpath+"/PAR1.png"
	ylabel = "Pressure"
	if ftype == "const" or ftype == "dcon":
		xlabel = "$\\alpha=f_0^2\\tau/T\\zeta$"
	elif ftype == "lin" or ftype == "lico" or ftype == "dlin":
		xlabel = "$\\alpha=k\\tau/\\zeta$"
	xlim = (AA[0],AA[-1])
	
	## ------------------------------------------------
	## Plot pressure
			
	## E2 prediction
	# if ftype == "lin" or ftype == "dlin":
		# plt.plot(AA,np.sqrt(1/(AA+1)),"b:",label="$(\\alpha+1)^{-1/2}$",lw=2)
	
	## Plot pressure against alpha for R or for S
	if ftype == "const" or ftype == "lin" or ftype == "linco":
		for i in range(RR.size):
			ax.plot(AA,PP[i,:],  "o-", label="$R = "+str(RR[i])+"$")
			
	elif ftype == "dcon" or ftype == "dlin":
		## Holding R fixed
		if RR.size == 1:
			for i in range(SS.size):
				ax.plot(AA,PP[i,0,:],  "o-", label="$R = "+str(RR[0])+",\,S = "+str(SS[i])+"$") 
				ax.plot(AA,QQ[i,0,:], "o--", color=ax.lines[-1].get_color())
		## Constant interval
		elif np.unique(RR-SS).size == 1:
			PP *= PP_WN; QQ *= QQ_WN
			title = "Pressure difference, $P_R-P_S$; $R-S = "+str((RR-SS)[0])+"$; ftype = "+ftype
			##
			# plotfile = dirpath+"/DPAR.png"
			# for i in range(RR.size):		## To plot against alpha
				# ax.plot(AA,np.diagonal(PP-QQ).T[i,:], "o-", label="$R = "+str(RR[i])+"$")
				# ax.plot(AA,np.diagonal(PP_WN-QQ_WN).T[i,:], "--", color=ax.lines[-1].get_color())
			plotfile = dirpath+"/DPRA.png"
			xlabel = "$R$"; ylabel = "Pressure Difference"; xlim = (RR[0],RR[-1])
			for i in range(AA.size/2):	## To plot against R
				ax.plot(RR,np.diagonal(PP-QQ).T[:,2*i], "o-", label="$\\alpha = "+str(AA[2*i])+"$") 
				ax.plot(RR,np.diagonal(PP_WN-QQ_WN).T[:,2*i], "--", color=ax.lines[-1].get_color()) 
			
	## ------------------------------------------------
	## Accoutrements
	
	ax.set_xlim(xlim)
	# ax.set_ylim(bottom=0.0)
	# ax.set_ylim(top=2.0)
	
	ax.set_xlabel(xlabel,fontsize=fsa)
	ax.set_ylabel(ylabel,fontsize=fsa)
	
	ax.grid()
	ax.legend(loc="best",fontsize=fsl)
	ax.set_title(title,fontsize=fst)
	
	plt.tight_layout()
	if not nosave:
		plt.savefig(plotfile)
		if verbose: print me+"plot saved to",plotfile
		
	return

##=============================================================================
def pdf_WN(r,fpars,ftype):
	"""
	Theoretical radial pdf of a white noise gas.
	"""
	if ftype is "const":
		R = fpars[0]
		Rind = np.argmin(np.abs(r-R))
		# rho0 = 1.0/(R+1.0)
		rho0 = 1.0/(0.5*R*R+R+1.0)
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.exp(R-r[Rind:])])
	elif ftype is "lin":
		R = fpars[0]
		Rind = np.argmin(np.abs(r-R))
		# rho0 = 1.0/(R+np.sqrt(np.pi/2))
		rho0 = 1.0/(0.5*R*R+np.sqrt(np.pi/2)*R+1.0)
		rho_WN = rho0 * np.hstack([np.ones(Rind),np.exp(-0.5*(r[Rind:]-R)**2)])
	elif ftype is "dcon":
		R, S = fpars
		Rind, Sind = np.argmin(np.abs(r-R)), np.argmin(np.abs(r-S))
		# rho0 = 1.0/(R-S+2-np.exp(-S))
		rho0 = 1.0/(S+R+np.exp(-S)+0.5*R*R-0.5*S*S)
		rho_WN = rho0 * np.hstack([np.exp(r[:Sind]-S),np.ones(Rind-Sind),np.exp(R-r[Rind:])])
	elif ftype is "dlin":
		R, S = fpars
		Rind, Sind = np.argmin(np.abs(r-R)), np.argmin(np.abs(r-S))
		# rho0 = 1.0/(R-S+np.sqrt(np.pi/2)*(1.0+sp.special.erf(S/np.sqrt(2))))
		rho0 = 1.0/(np.exp(-0.5*S*S)+np.sqrt(np.pi/2)*S*sp.special.erf(S/np.sqrt(2))+0.5*R*R-0.5*S*S+np.sqrt(np.pi/2)*R)
		rho_WN = rho0 * np.hstack([np.exp(-0.5*(S-r[:Sind])**2),np.ones(Rind-Sind),np.exp(-0.5*(r[Rind:]-R)**2)])
	return rho_WN

def Hr_norm(H,r,R):
	"""
	H is probability density per unit area (flat in the bulk).
	Hr is probability density.
	"""
	# H[0]=H[1]
	rext = np.hstack([np.linspace(0.0,r[0],2),r])
	Hext = np.hstack([H[:np.argmin(np.abs(r-R))/2].mean()*np.ones(2),H])
	# H /= np.trapz(rext*Hext,x=rext)
	H /= np.trapz(Hext,x=rext)
	return H

	
##=============================================================================
if __name__=="__main__":
	main()
