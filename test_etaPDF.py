me0 = "test_etaPDF"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv
import os
from LE_Utils import filename_par

"""
Plot pdf of eta if given a file. pdf split into three regions.
"""

try:

	histfile = argv[1]

	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	
	inner = False if S==0.0 else True
	bulk = False if R==S else True
	
	normalise = False
	
	try: nosave = not bool(argv[2])
	except IndexError: nosave = True

	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	rmax = rbins[-1]
	r = 0.5*(rbins[1:]+rbins[:-1])
	erbins = bins["erbins"]
	er = 0.5*(erbins[1:]+erbins[:-1])
	
	## Wall indices
	Rind = np.abs(r-R).argmin()
	Sind = np.abs(r-S).argmin()
		
	## Load histogram; normalise
	H = np.load(histfile)
	try:	H = H.sum(axis=2)
	except ValueError:	pass
	H /= np.trapz(np.trapz(H,er,axis=1),r,axis=0)
	
	## Distribution on either side of the wall: inner, outer, bulk
	rR = r[Rind:];		HR = H[Rind:,:]
	if inner:	rS = r[:Sind];		HS = H[:Sind,:]
	if bulk:	rB = r[Sind:Rind];	HB = H[Sind:Rind,:]
	## After the split, position is irrelevant
	H  = np.trapz(H, r, axis=0)
	HR = np.trapz(HR, rR, axis=0)
	if inner:	HS = np.trapz(HS, rS, axis=0)
	if bulk:	HB = np.trapz(HB, rB, axis=0)
	## rho is probability density
	erho  = H/(2*np.pi*er)
	erhoR = HR/(2*np.pi*er)
	if inner:	erhoS = HS/(2*np.pi*er)
	if bulk:	erhoB = HB/(2*np.pi*er)
	
	## Normalise each individually
	if normalise:
		erho /= np.trapz(H, er)
		erhoR /= np.trapz(HR, er)
		if inner:	erhoS /= np.trapz(HS, er)
		if bulk: erhoB /= np.trapz(HB, er)
		suf = "_norm"
	else:
		suf = ""

	##---------------------------------------------------------------			
	## PLOT SET-UP
		
	figtit = "Density; $\\alpha="+str(a)+"$"
	fig, ax = plt.subplots(1,1)
	fsa = 16
			
	##---------------------------------------------------------------	
	## PDF PLOT
	
	ax.plot(er, erho, "k-", label="Total")
	ax.fill_between(er,0,erho,facecolor="black",alpha=0.1)
	ax.plot(er, erhoR, "g-", label=r"Outer $r>R$")
	ax.fill_between(er,0,erhoR,facecolor="green",alpha=0.1)
	if inner:
		ax.plot(er, erhoS, "r-", label=r"Inner $S>r$")
		ax.fill_between(er,0,erhoS,facecolor="red",alpha=0.1)
	if bulk:
		ax.plot(er, erhoB, "b-", label="Bulk $R>r>S$")
		ax.fill_between(er,0,erhoB,facecolor="blue",alpha=0.1)

	## Fit
	er2 = np.linspace(er[0],er[-1]*(1+0.5/er.size),er.size*5+1)
	fitfunc = lambda x, A, s2:\
				A/(2*np.pi*s2)*np.exp(-0.5*x*x/s2)
	fitT = sp.optimize.curve_fit(fitfunc, er, erho, p0=[1.0,1.0])[0]
	fitR = sp.optimize.curve_fit(fitfunc, er, erhoR, p0=[1.0,1.0])[0]
	fitS = sp.optimize.curve_fit(fitfunc, er, erhoS, p0=[1.0,1.0])[0] if inner else [0,0]
	fitB = sp.optimize.curve_fit(fitfunc, er, erhoB, p0=[1.0,1.0])[0] if bulk else [0,0]
	print [a,R,S],"\t s2 = ",np.around([fitT[1], fitB[1], fitR[1], fitS[1]],3)

	## Accoutrements
	ax.set_xlim(left=0.0)
	ax.set_xlabel(r"$\eta$", fontsize=fsa)
	ax.set_ylabel(r"$\rho(\eta,\phi)$", fontsize=fsa)
	ax.grid()
	ax.legend()

	plt.title(r"PDF of $\eta$, divided into regions. $\alpha="+str(a)+"$, $R="+str(R)+"$, $S="+str(S)+"$")
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFeta"+os.path.basename(histfile)[4:-4]+suf+".jpg"
		fig.savefig(plotfile)
		print me0+": Figure saved to",plotfile
	# plt.show()
	exit()

## =========================================================================================

except IndexError:
	
	## Cannot remember where this data is from...
	# a = np.array([0.2,0.4,0.5,0.7,1.0,1.5,2.0,2.5,3.0,5.0,10.0])
	# s2 = np.array([5.2515,2.5577,2.0466,1.4397,1.0110,0.6710,0.5061,0.3991,0.3349,0.2009,0.1006])

	plt.plot(a, s2, "bo-", label=r"$\sigma^2$")
	plt.plot(a, a**(-1.), "k--", label=r"$1/\alpha$")

	plt.xlabel(r"$\alpha$")
	plt.xscale("log")
	plt.yscale("log")
	plt.legend(loc="best")
	plt.grid()

	plt.show()