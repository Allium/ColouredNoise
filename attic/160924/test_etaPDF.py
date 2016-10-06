import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv
import os
from LE_Utils import filename_par

"""
Plot pdf of eta if given a file.
Otherwise make plot of s2 versus alpha from previously collected data (160917).
"""

try:

	histfile = argv[1]

	a = filename_par(histfile, "_a")

	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	rmax = rbins[-1]
	r = 0.5*(rbins[1:]+rbins[:-1])
	erbins = bins["erbins"]
	er = 0.5*(erbins[1:]+erbins[:-1])
		
	## Load histogram, convert to normalised pdf
	H = np.load(histfile)
	try:	H = H.sum(axis=2)
	except ValueError:	pass
	## Position irrelevant here
	H = np.trapz(H, r, axis=0)
	## rho is probability density
	erho = H/(2*np.pi*er) / np.trapz(H, er)

	##---------------------------------------------------------------			
	## PLOT SET-UP
		
	figtit = "Density; $\\alpha="+str(a)+"$"
	fig, ax = plt.subplots(1,1)
	fsa = 16
			
	##---------------------------------------------------------------	
	## PDF PLOT

	ax.plot(er,erho, "b-", label="CN simulation")

	## Fit TEMPORARY
	er2 = np.linspace(er[0],er[-1]*(1+0.5/er.size),er.size*5+1)
	fitfunc = lambda x, A, s2:\
				A/(2*np.pi*s2)*np.exp(-0.5*(x)**2./s2)
	fit = sp.optimize.curve_fit(fitfunc, er, erho, p0=[1.0,1.0])[0]
	print [a],"\t[A, B] = ",fit
	ax.plot(er2, fitfunc(er2, *fit), "g--", lw=2)

	## Accoutrements
	ax.set_xlim(left=0.0)
	ax.set_xlabel(r"$\eta$", fontsize=fsa)
	ax.set_ylabel(r"$\rho(\eta,\phi)$", fontsize=fsa)
	ax.grid()

	plt.title(r"$\alpha="+str(a)+"$")
	plt.show()
	exit()

## =========================================================================================

except IndexError:

	a = np.array([0.2,0.4,0.5,0.7,1.0,1.5,2.0,2.5,3.0,5.0,10.0])

	s2 = np.array([5.25150854,2.55776206,2.04659908,1.43965591,1.01097344,0.67096949,0.50612912,0.39905598,0.33493162,0.20093551,0.10061558])

	plt.plot(a, s2, "bo-", label=r"$\sigma^2$")
	plt.plot(a, a**(-1.), "k--", label=r"$1/\alpha$")

	plt.xlabel(r"$\alpha$")
	plt.xscale("log")
	plt.yscale("log")
	plt.legend(loc="best")
	plt.grid()

	plt.show()