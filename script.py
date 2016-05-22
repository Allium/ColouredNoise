import numpy as np
import os, glob, time
from sys import argv


t00 = time.time()


##=============================================================================
## 1D
if argv[1]=="1D":
	from LE_LightBoundarySim import *
	for a in [3.0]:# np.arange(0.0,10.1,0.2):
		for X in [0.0]:
			t0=time.time()
			print "\n\nscript: [a, X] =",np.around([a,X],2)
			tmax = 12*max(a/4,0.5)
			main(a,"const",X,0.0,1,0.01,tmax,True)
			#main(a,"lin",X,0.0,1,0.01,tmax,True)
			print "script: [a, X] =",np.around([a,X],2),"execution time",round(time.time()-t0,2),"seconds"

##=============================================================================
## DISC

from LE_SBS import *

## SINGLE
if argv[1]=="D":
	for R in [2.0]:
		for a in [0.4,2.0]:#np.hstack([np.arange(0.0,2.1,0.2),np.arange(2.4,3.7,0.4)]):
			for lam in [0.1]:
				for nu in [10.0]:
					t0=time.time()
					print "\n\nscript: [a, R, l, n] =",np.around([a,R,lam,nu],2)
					tmax = 10*max(a/4,0.4)
					main(a,"const",[R,0.0,lam,nu],1,0.01,tmax,True)
					print "script: [a, R, l, n] =",np.around([a,R,lam,nu],2),"execution time",round(time.time()-t0,2),"seconds"


## DOUBLE
if argv[1]=="DD":
	for a in np.hstack([np.arange(0.2,2.1,0.2),np.arange(2.4,3.7,0.4)]):
		for R in [2.0,5.0]:
			for S in [R-1.0]:
				for lam in [0.5]:
					for nu in [10.0]:
						t0=time.time()
						tmax = 10*max(a/4,0.4)
						print "script: [a, R, S, l] =",np.around([a,R,S,lam],2)
						main(a,"dlin",[R,S,lam,nu],1,0.01,tmax,True)
						main(a,"dcon",[R,S,lam,nu],1,0.01,tmax,True)
						print "script: [a, R, S, l] =",np.around([a,R,S,lam],2),"execution time",round(time.time()-t0,2),"seconds"

##=============================================================================

print "\nscript: total time",round(time.time()-t00,2),"seconds"
