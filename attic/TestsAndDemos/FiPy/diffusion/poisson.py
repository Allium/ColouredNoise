## http://www.pritschet.me/wiki/python/fipy/fipy-solving-pdes-python/

#!/usr/bin/env python

from fipy import Grid2D, CellVariable, TransientTerm, DiffusionTerm, FixedValue
import pylab
import sys

nx = 300
dx = 0.05
L = nx * dx
mesh = Grid2D(dx=dx,dy=dx,nx=nx,ny=nx)
x,y = mesh.getCellCenters()
x0,y0 = L/2,L/2
X,Y = mesh.getFaceCenters()

potential = CellVariable(mesh=mesh, name='potential', value=0.)
potential.equation = (DiffusionTerm(coeff = 1.) == 0.)

bcs = (
    FixedValue(value=5,faces=mesh.getFacesLeft() & (Y<y0) ),
    FixedValue(value=0,faces=mesh.getFacesRight() & (Y<y0) ),
    FixedValue(value=2,faces=mesh.getFacesTop() ),
)

potential.equation.solve(var=potential, boundaryConditions=bcs)

# The follow evaluation of the solution is only
# possible with "common" meshes.
result = pylab.array(potential)
result = result.reshape((nx,nx))
xx,yy = pylab.array(x), pylab.array(y)
xx,yy = xx.reshape((nx,nx)), yy.reshape((nx,nx))

pylab.title("Capacitor: +5/+0 V, Top Plate: +2 V")
pylab.contourf(xx,yy,result,levels=pylab.linspace(result.min(),result.max(),64))
pylab.colorbar()
pylab.show()