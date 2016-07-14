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

from LE_SBS import *

##=============================================================================

## DISC
if argv[1]=="D":
	for R in [0.0]:
		for lam in [0.5]:
			for a in [0.4,0.6,0.8,1.2,2.0,1.6]:#np.hstack([np.arange(0.0,2.1,0.2),np.arange(2.4,3.7,0.4)]):
				for nu in [5.0]:
					for intmeth in ["eul"]:
						t0=time.time()
						print "\n\nscript: [a, R, l, n] =",np.around([a,R,lam,nu],2)
						ephi = False
						dt = 0.01
						tmax = 4*max(a*a/4.0,0.4)
						#main(a,"const",[R,0.0,lam,nu],1,dt,tmax,intmeth,ephi,True)
						main(a,"lin",[R,0.0,lam,nu],1,dt,tmax,intmeth,ephi,True)
						#main(a,"tan",[R,0.0,lam,nu],1,dt,tmax,intmeth,ephi,True)
						# main(a,"nu",[R,0.0,lam,nu],1,dt,tmax,intmeth,ephi,True)
						print "script: [a, R, l, n] =",np.around([a,R,lam,nu],2),"execution time",round(time.time()-t0,2),"seconds"

##=============================================================================

## ANNULUS
if argv[1]=="DD":
	for R in [20.0]:
		for S in [R-10.0]:
			for lam in [1.0]:
				for ftype in ["dlin"]:
					for a in [1.0]:
						t0=time.time()
						print "script: [a, R, S, l] =",np.around([a,R,S,lam],2)
						ephi = True; nu = 1.0; intmeth = ""
						dt = 0.005 if a<=1.0 else 0.01
						tfac = 60
						tmax = tfac*max(a*a/4,0.4)
						main(a,ftype,[R,S,lam,nu],1,dt,tmax,intmeth,ephi,True)
						print "script: [a, R, S, l] =",np.around([a,R,S,lam],2),"execution time",round(time.time()-t0,2),"seconds"


##=============================================================================

print "\nscript: total time",round(time.time()-t00,2),"seconds"
