import numpy as np
from matplotlib import pyplot as plt
from LE_Utils import fs

fsa,fsl,fst=fs


S = np.array([0.0,1.0,2.0,5.0,10.0,20.0,50.0,100.0])
nuO = [0.0,0.248,0.265,0.305,0.367,0.422,0.468,0.481]
nuI = [np.nan,0.848,0.910,0.795,0.670,0.591,0.536,0.519]

fig, ax = plt.subplots(1,1)

ax.plot(S, nuO, "o-", label=r"$P_{\rm out}$")
ax.plot(S, nuI, "o-", label=r"$P_{\rm in}$")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$R=S$", fontsize=fsa)
ax.set_ylabel(r"$\nu$", fontsize=fsa)
ax.set_title(r"Exponent for $P(\alpha)\sim(1+\alpha)^\nu$. $R=S$.", fontsize=fst)
ax.grid()
ax.legend(fontsize=fsl)

plt.show()
