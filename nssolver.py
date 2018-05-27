from dolfin import *
import numpy as np

# Use UFLACS to speed-up assembly and limit quadrature degree
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
#parameters['form_compiler']['quadrature_degree'] = 4

def flow_around_cylinder((mesh,bndry)):

    # Define finite elements
    Ep = FiniteElement("CG",mesh.ufl_cell(),1)
    Ev = VectorElement("CG",mesh.ufl_cell(),2)
    
    # Build function spaces (Taylor-Hood)
    W = FunctionSpace(mesh,MixedElement([Ev, Ep]))
    
   # No-slip boundary condition for velocity on walls and cylinder - boundary id 3
    noslip = Constant((0, 0))
    bcv_walls = DirichletBC(W.sub(0), noslip, bndry, 212)

    v_in=Constant((0.1,0))
    bcv_in = DirichletBC(W.sub(0), v_in, bndry, 204)    #in
    bcv_out = DirichletBC(W.sub(0), v_in, bndry, 205)   #out
    bcv_wall1 = DirichletBC(W.sub(0), v_in, bndry, 206) #horni 
    bcv_wall2 = DirichletBC(W.sub(0), v_in, bndry, 207) #dolni


    # Collect boundary conditions
    bcs =  [bcv_walls, bcv_in,bcv_out,bcv_wall1,bcv_wall2]
   
    # Facet normal, identity tensor and boundary measure
    n = FacetNormal(mesh)
    I = Identity(mesh.geometry().dim())
    ds = Measure("ds", subdomain_data=bndry)
    nu = Constant(0.01)

    v, q = TestFunctions(W)
    w = Function(W)
    u, p = split(w)

    # Define variational forms
    T = -p*I + 2.0*nu*sym(grad(u))
    F = inner(T, grad(v))*dx - q*div(u)*dx + inner(grad(u)*u, v)*dx

    begin("Solving problem of size: {0:d}".format(W.dim()))
    problem=NonlinearVariationalProblem(F,w,bcs,derivative(F,w))
    solver=NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1E-12
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-12
    solver.parameters['newton_solver']['maximum_iterations'] = 20
    solver.solve()
    end()
    
    # Report drag and lift
    force = dot(T, n)
    D = -(force[0]/0.002)*ds(212)
    L = -(force[1]/0.002)*ds(212)
    drag = assemble(D)
    lift = assemble(L)

   # Alternative force calculation
#    w_=Function(W)
#    DirichletBC(W.sub(0),(1.0,0.0),bndry,212).apply(w_.vector())
#    drag=-assemble(action(F,w_))/0.002

#    w_=Function(W)
#    DirichletBC(W.sub(0),(0.0,1.0),bndry,212).apply(w_.vector())
#    lift=-assemble(action(F,w_))/0.002
                                    
    info("drag= {0:e}    lift= {1:e}".format(drag , lift))

    return(w, W.dim(),drag,lift)


if __name__ == "__main__":

   # import mymesh
   # m = mymesh.generate(0)
    mesh=Mesh('NACA0012-10.xml')
    bndry = MeshFunction("size_t", mesh, 'NACA0012-10_facet_region.xml')
    result=flow_around_cylinder((mesh,bndry))
    
    plot(result[0].sub(0), title='velocity')
    plot(result[0].sub(1), title='pressure')
    interactive()
