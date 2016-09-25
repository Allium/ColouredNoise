import numpy as np
from matplotlib import pyplot as plt

"""
Plot offsets from expected peak.
Data from fits to files in 160612_CIR_DL_dt0.01
"""

A = [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.6,2.0]
R = [0.0,0.5,1.0,2.0,3.0,4.0,5.0]

offR0n5 = [-0.00149, 0.004295, -0.01088, 0.000909, -0.02185, -0.00214, 0.000791, -0.00166, -0.001134]
offR0n10 = [0.000785,-0.00099,-0.00248,-0.00347,0.001451,-0.00149,-0.00294, -0.00349, -0.00067]

offR05n5 = [-0.03414,0.136633,0.138515,0.135818,0.128821,0.127508,0.122767,0.115324,0.108658]
offR05n10 = [-0.0300, 0.08784, 0.08477, 0.08064, 0.07518, 0.07296, 0.06887, 0.06516, 0.06031]

offR1n5 = [-0.00917,0.089111,0.103729,0.103626,0.102877,0.102690,0.101002,0.096997,0.092315]
offR1n10 = [-0.00706,0.058164,0.061337,0.059840,0.058544,0.056433,0.055174,0.052834,0.050178]

offR2n10 = [-0.002515,0.0319393,0.0379985,0.0383159,0.038082,0.03809,0.038308,0.0371006,0.035503]
offR2n5 = [-0.000844, 0.0503131, 0.062517, 0.0642637, 0.066868, 0.070970, 0.068205, 0.0679489, 0.066361]

offR3n10 = [-0.00234,0.022682,0.024834,0.026701,0.027977,0.029152,0.028695,0.027095,0.027094]
offR3n5 = [-0.0021,0.03518,0.04426,0.04737,0.04734,0.04840,0.05111,0.05284,0.05091]

offR4n10 = [-0.0008,0.01567,0.02025,0.02168,0.02243,0.02111,0.02232,0.02198,0.02108]
offR4n5 = [-0.00065,0.028378,0.034242,0.037864,0.036064,0.039911,0.040138,0.039862,0.03976]

offR5n10 = [-0.00216,0.015729,0.015547,0.017459,0.015984,0.018062,0.018816,0.017895,0.017375]
offR5n5 = [-0.0013,0.02412,0.02711,0.02932,0.02656,0.03305,0.03162,0.03344,0.03287]

# offR10n5 = np.zeros(len(A))
# offR10n10 = [0.0033340,0.0033333,0.0033333,0.0033333,0.0033334,0.0033333,0.0033341,0.0033341,0.0033333]


fig, axs = plt.subplots(1,2,sharey=True)


ax = axs[0]
ax.plot(A,offR0n10,"o-", label="R0")
# ax.plot(A,offR0n5,"o--", c=ax.lines[-1].get_color(), label="R0 5")
ax.plot(A,offR05n10,"o-", label="R05")
ax.plot(A,offR1n10,"o-", label="R1")
# ax.plot(A,offR1n5,"o--", c=ax.lines[-1].get_color(), label="R2 5")
ax.plot(A,offR2n10,"o-", label="R2")
ax.plot(A,offR3n10,"o-", label="R2")
ax.plot(A,offR4n10,"o-", label="R4")
ax.plot(A,offR5n10,"o-", label="R5")
# ax.plot(A,offR5n5,"o--", c=ax.lines[-1].get_color(), label="R5 5")

ax.grid()
ax.legend(loc="lower right", fontsize=8)
ax.set_xlabel("$\\alpha$",fontsize=16)
ax.set_ylabel("Peak offset",fontsize=16)


ax=axs[1]
trandat5 = zip(A,offR0n5,offR05n5,offR1n5,offR2n5,offR3n5,offR4n5,offR5n5)
trandat10 = zip(A,offR0n10,offR05n10,offR1n10,offR2n10,offR3n10,offR4n10,offR5n10)
for i,a in enumerate(A):
	ax.plot(R,trandat10[i][1:], "o-", label="a"+str(a))
	# ax.plot(R,trandat5[i][1:], "o--", c=ax.lines[-1].get_color(), label="a"+str(a)+" 5")
# ax.set_yscale("log");# ax.set_xscale("log")
	
ax.grid()
ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel("$R$",fontsize=16)


fig.suptitle("Offset. Solid $\\nu=10$, dashed $\\nu=5$.")
plt.show()