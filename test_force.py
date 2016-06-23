import numpy as np
from matplotlib import pyplot as plt
from time import time

from LE_SBS import force_nu, force_dnu,\
					fxy_infpot
from LE_Utils import plot_fontsizes


plt.rcParams.update({"axes.labelsize": plot_fontsizes()[0]})

R = 2.0
S = 1.0
lam = 0.5
wob = R+lam
wib = S-lam
nu = 10.0
dt = 0.01

force = lambda xy, r: force_dnu(xy,r,R,S,lam,nu)

## FORCE MAGNITUDE

#r = np.arange(0.01,wob+1.0,0.01)
#f = np.array([fxy_infpot(ri,ri*ri, force, wob,wib,dt) for ri in r])

#plt.plot(r,f)
#plt.axvline(S,c="r",ls="--",lw=2.0); plt.axvline(R,c="y",ls="--",lw=2.0)
#plt.axvline(wib,c="r",ls="-",lw=2.0); plt.axvline(wob,c="y",ls="-",lw=2.0)

##plt.ylim([-100,100])
#plt.grid()
#plt.xlabel("$r$"); plt.ylabel("$f(r)$")
#plt.show()
#plt.close()


## FORCE DIRECTION

ang = np.linspace(0.0,2*np.pi,360)
plt.plot(R*np.cos(ang),R*np.sin(ang),"y--",(R+lam)*np.cos(ang),(R+lam)*np.sin(ang),"y-",lw=2.0)
plt.plot(S*np.cos(ang),S*np.sin(ang),"r--",(S-lam)*np.cos(ang),(S-lam)*np.sin(ang),"r-",lw=2.0)

x = np.linspace(-3.0,3.0,31)
y = np.linspace(-3.0,3.0,31)

X,Y = np.meshgrid(x,y)
RR = np.sqrt(X*X+Y*Y)
F = force([X,Y],RR)
F[:,RR>wob]=0.0
F[:,RR<wib]=0.0

#radvec = np.array([(xi, yi) for xi in x for yi in y]).reshape((x.size,y.size,2)).T
#plt.quiver(X,Y, radvec[0], radvec[1])
#plt.show(); exit()

#F = np.array([fxy_infpot(np.array([xi,yi]),np.sqrt(xi*xi+yi*yi)+0.001, force, wob,wib,dt)\
#				for xi in x for yi in y]).reshape((x.size,y.size,2)).T
				
plt.plot(X,Y,"ko",markersize=1.0)
plt.quiver(X,Y, F[0], F[1], scale=800)

plt.grid()
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.show()



