me0 = "schem_flow"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import NullLocator
from LE_Utils import fs, set_mplrc
set_mplrc(fs)

ax = plt.axes()

## xy
ax.arrow(-1.0, 0.0, 2.0, 0.0, head_width=0.05, head_length=0.1, length_includes_head=True, overhang=0.5, fc='k', ec='k')
ax.arrow(0.0, -1.0, 0.0, 2.0, head_width=0.05, head_length=0.1, length_includes_head=True, overhang=0.5, fc='k', ec='k')
ax.text(0.92, -0.10, r"$x$", fontsize=fs["fsa"])
ax.text(+0.05, 0.92, r"$\eta$", fontsize=fs["fsa"])

## Bulk arrows
xb = 0.5
num_bulk_arrow = 7
for i in range(1,num_bulk_arrow+1):
	x = i*(xb/(num_bulk_arrow+1))
	y = i*(1.0/(num_bulk_arrow+1))
	print [x,y]
	ax.arrow(-x, +y, +2*x, 0.0, head_width=0.05, head_length=0.05, length_includes_head=True, fc='b', ec='b')
	ax.arrow(+x, -y, -2*x, 0.0, head_width=0.05, head_length=0.05, length_includes_head=True, fc='b', ec='b')

arr = FancyArrowPatch(posA=[0,1], posB=[0,0], path=None, arrowstyle='simple', arrow_transmuter=None, connectionstyle='arc3', connector=None, patchA=None, patchB=None, shrinkA=2.0, shrinkB=2.0, mutation_scale=1.0, mutation_aspect=None, dpi_cor=1.0)	
arr.set_connectionstyle("arc,angleA=0,armA=30,rad=10")
ax.add_patch(arr)

# common_opts = dict(arrowstyle=u'->', lw=3)
# arrow_patch_0 = FancyArrowPatch(posA=(0.2, 0.8), posB=(0.8, 0.65),
                                # mutation_scale=50, **common_opts)
# arrow_patch_1 = FancyArrowPatch(posA=(0.2, 0.2), posB=(0.8, 0.45),
                                # mutation_scale=150, **common_opts)

# ax.text(0.2, 0.85, "mutation_scale = 50", ha='left', va='bottom')
# ax.text(0.2, 0.15, "mutation_scale = 150", ha='left', va='top')
# for arrow_patch in [arrow_patch_0, arrow_patch_1]:
    # ax.add_patch(arrow_patch)



## Axis limits
ax.set_xlim(-1,+1)
ax.set_ylim(-1,+1)

## Axis labels
ax.xaxis.set_major_locator(NullLocator())
ax.yaxis.set_major_locator(NullLocator())

# ax.axis("off")
# ax.patch.set_visible(False)
plt.gcf().patch.set_visible(False)

plt.show()