me0 = "test_coord"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator
from LE_Utils import fs, set_mplrc
set_mplrc(fs)

# fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
ax = plt.axes()

## xy
ax.arrow(-0.1, 0.0, 1.0, 0.0, head_width=0.05, head_length=0.1, length_includes_head=True, overhang=0.5, fc='k', ec='k')
ax.arrow(0.0, -0.1, 0.0, 1.0, head_width=0.05, head_length=0.1, length_includes_head=True, overhang=0.5, fc='k', ec='k')
ax.text(0.95, 0.0, r"$x$", fontsize=fs["fsa"])
ax.text(0.0, 0.95, r"$y$", fontsize=fs["fsa"])

## r and eta
r = [0.8, 0.4]
e = [1.2, 1.2]
ax.arrow(0.0, 0.0, r[0], r[1], head_width=0.05, head_length=0.1, length_includes_head=True, fc='b', ec='b')
ax.text(0.5*r[0], 0.6*r[1], r"$\vec r$", fontsize=fs["fsa"], color="b")
ax.arrow(r[0], r[1], e[0]-r[0], e[1]-r[1], head_width=0.05, head_length=0.1, length_includes_head=True, fc='r', ec='r')
ax.text(0.46*(e[0]+r[0]), 0.5*(e[1]+r[1]), r"$\vec \eta$", fontsize=fs["fsa"], color="r")

## Construction lines
fac = 1.6
ax.plot([r[0],fac*r[0]],[r[1],fac*r[1]], lw=2, ls="--", c="b")
ax.plot([r[0],fac*r[0]],[r[1],r[1]], lw=2, ls="--", c="k")

## Angle arcs
p = np.linspace(0.0,np.arctan2(r[1],r[0]),51)
R = 0.5
ax.plot(R*np.cos(p),R*np.sin(p),"b")
ax.text(R, 0.2*r[1], r"$\chi$", fontsize=fs["fsa"], color="b")
p = np.linspace(0.0,np.arctan2(e[1]-r[1],e[0]-r[0]),51)
R = 0.2
ax.plot(r[0]+R*np.cos(p),r[1]+R*np.sin(p),"r")
ax.text(r[0]+R-0.05, r[1]+0.2*(e[1]-r[1]), r"$\phi$", fontsize=fs["fsa"], color="r")

p = np.linspace(np.arctan2(r[1],r[0]),np.arctan2(e[1]-r[1],e[0]-r[0]),51)
R = 0.4
ax.plot(r[0]+R*np.cos(p),r[1]+R*np.sin(p),"k")
ax.text(r[0]+R-0.12, r[1]+0.5*(e[1]-r[1])-0.10, r"$\psi$", fontsize=fs["fsa"], color="k")

## Particle
ax.plot(r[0],r[1], "yo", ms=20)

## Axis limits
m = max(r[0]*fac,e[0],e[1])
ax.set_xlim(-0.1,m)
ax.set_ylim(-0.1,m)

## Axis labels
ax.xaxis.set_major_locator(NullLocator())
ax.yaxis.set_major_locator(NullLocator())

ax.axis("off")
ax.patch.set_visible(False)
plt.gcf().patch.set_visible(False)

plt.show()