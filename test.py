from dolfin import *
import numpy as np
import nssolver

name="NACA2412"
meshes=[]
# create sequence of 4 refined meshes
for i in range (0,22,2):
    mesh=Mesh('NACA2412/NACA2412-{}.xml'.format(i))
    bndry = MeshFunction("size_t", mesh, 'NACA2412/NACA2412-{}_facet_region.xml'.format(i))
    meshes.append((mesh,bndry))

# open files for paraview solutions
ufile = XDMFFile("results_{0:s}/u.xdmf".format(name))
pfile = XDMFFile("results_{0:s}/p.xdmf".format(name))

# Call solver for all meshes
results2412 = [ nssolver.flow_around_cylinder(m) for m in meshes ]

solutions = zip(*results2412)[0]
results2412=np.array(zip(*results2412)[1:]).T

# Save solution for paraview inspection
for w in solutions :
    u, p = w.split()
    u.rename("u", "velocity")
    p.rename("p", "pressure")
    ufile.write(u)
    pfile.write(p)

name="NACA0012"
meshes=[]
# create sequence of 4 refined meshes
for i in range (0,22,2):
    mesh=Mesh('NACA0012/NACA0012-{}.xml'.format(i))
    bndry = MeshFunction("size_t", mesh, 'NACA0012/NACA0012-{}_facet_region.xml'.format(i))
    meshes.append((mesh,bndry))

# open files for paraview solutions
ufile = XDMFFile("results_{0:s}/u.xdmf".format(name))
pfile = XDMFFile("results_{0:s}/p.xdmf".format(name))

# Call solver for all meshes
results = [ nssolver.flow_around_cylinder(m) for m in meshes ]

solutions = zip(*results)[0]
results=np.array(zip(*results)[1:]).T

# Save solution for paraview inspection
for w in solutions :
    u, p = w.split()
    u.rename("u", "velocity")
    p.rename("p", "pressure")
    ufile.write(u)
    pfile.write(p)




import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot( [i for i in range(0,22,2)],results2412[:,1], 'x-', label='drag-2412')
plt.plot( [i for i in range(0,22,2)],results2412[:,2], 'o-', label='lift-2412')
plt.plot( [i for i in range(0,22,2)],results[:,1], 'x-', label='drag-0012')
plt.plot( [i for i in range(0,22,2)],results[:,2], 'o-', label='lift-0012')
plt.title('Flow around NACA')
plt.xlabel('Angle of elevation')
plt.ylabel('Force')
plt.legend(loc=2, prop={'size': 10})
plt.savefig('graph.pdf'.format(name), bbox_inches='tight')
