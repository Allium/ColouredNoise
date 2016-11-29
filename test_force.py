import os, time
import timeit
import numpy as np
import scipy as sp
import scipy.integrate
from matplotlib import pyplot as plt
from LE_Utils import fs

from LE_CSim import force_clin, force_mlin, force_nlin
from LE_SSim import force_dlin


R = 2.0
S = 1.0
T = 0.0
	
"""
x = np.linspace(-3*R,3*R,1000)
fx = force_mlin([x,0],R,S,T)[0]
fx = np.array([force_mlin([xi,0],R,S,T) for xi in x])[:,0]

U = -sp.integrate.cumtrapz(fx,x,initial=0.0); U-=U.min()

fig, ax = plt.subplots(1,1, figsize=fs["figsize"])

ax.plot(x, fx, "-", label=r"$f(x)$", lw=3)
ax.plot(x, U, "-", label=r"$U(x)$", lw=3)

ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
#ax.set_ylabel(r"$f,U$", fontsize=fs["fsa"])
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.grid()
ax.legend(loc="best",fontsize=fs["fsl"]).get_frame().set_alpha(0.5)

plt.show()
exit()
"""

N = 10000
xy = np.linspace(0.0,1.0,10000).reshape((2,5000))
T1, T2, T3 = np.zeros((3,N))
for i in range(N):
	t0 = time.time()
	np.array([force_dlin(xyi,np.sqrt((xyi*xyi).sum()),R,S) for xyi in xy])[:,0]
	T1[i] = time.time()-t0
for i in range(N):
	t0 = time.time()
	np.array([force_dlin2(xyi,np.sqrt((xyi*xyi).sum()),R,S) for xyi in xy])[:,0]
	T2[i] = time.time()-t0
print "1 %.3g"%(T1.mean())
print "2 %.3g"%(T2.mean())
	
	
exit()
