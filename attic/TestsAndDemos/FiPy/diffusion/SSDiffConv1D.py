## http://www.ctcms.nist.gov/fipy/examples/convection/generated/examples.convection.exponential1D.mesh1D.html

from fipy import *
from matplotlib import pyplot as plt

diffCoeff = 1.
convCoeff = (10.,)

L = 10.
nx = 100
mesh = Grid1D(dx=L / nx, nx=nx)

valueLeft = 0.
valueRight = 1.

var = CellVariable(mesh=mesh, name="variable")

var.constrain(valueLeft, mesh.facesLeft)
var.constrain(valueRight, mesh.facesRight)

eq = (DiffusionTerm(coeff=diffCoeff) + ExponentialConvectionTerm(coeff=convCoeff))

eq.solve(var=var)

axis = 0
x = mesh.cellCenters[axis]
CC = 1. - numerix.exp(-convCoeff[axis] * x / diffCoeff)
DD = 1. - numerix.exp(-convCoeff[axis] * L / diffCoeff)
analyticalArray = CC / DD
print var.allclose(analyticalArray)

plt.plot(var)
plt.show()

