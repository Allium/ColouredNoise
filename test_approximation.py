import os, time
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from LE_Utils import filename_par, fs
from LE_CSim import force_nlin
from LE_PDFxy import calc_Q_NLsmallR


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



"""
R = 2.0
S = 1.0
T = 0.0

N = 5
x = np.linspace(0.0,1.0,10000)

for i in range(N):
	t0 = time.time()
	np.array(force_mlin2([x,0],R,S,T))[0]
	print "1 ",time.time()-t0
for i in range(N):
	t0 = time.time()
	np.array([force_mlin2([xi,0],R,S,T) for xi in x])[0]
	print "2 ",time.time()-t0
for i in range(N):
	t0 = time.time()
	np.array([force_mlin([xi,0],R,S,T) for xi in x])[0]
	print "3 ",time.time()-t0
"""
