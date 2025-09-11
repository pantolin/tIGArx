# class ExtractedNonlinearProblem(NonlinearProblem):
#     """
#     Class encapsulating a nonlinear problem posed on an extracted spline, to
#     allow existing nonlinear solvers (e.g., PETSc SNES) to be used.

#     NOTE: Obtaining the initial guess for the IGA DoFs from the given
#     FE function for the solution fields currently requires
#     a linear solve, which is performed using the spline object's solver,
#     if any.
#     """

#     def __init__(self, spline, residual, tangent, solution, **kwargs):
#         """
#         The argument ``spline`` is an ``ExtractedSpline`` on which the
#         problem is solved.  ``residual`` is the residual form of the problem.
#         ``tangent`` is the Jacobian of this form.  ``solution`` is a
#         ``Function`` in ``spline.V``.  Additional keyword arguments will be
#         passed to the superclass constructor.
#         """
#         super(ExtractedNonlinearProblem, self).__init__(**kwargs)
#         self.spline = spline
#         self.solution = solution
#         self.residual = residual
#         self.tangent = tangent

#     # Override methods from NonlinearProblem to perform extraction:
#     def form(self, A, P, B, x):
#         self.solution.vector()[:] = self.spline.M * x

#     def F(self, b, x):
#         b[:] = self.spline.assembleVector(self.residual)
#         return b

#     def J(self, A, x):
#         M = self.spline.assembleMatrix(self.tangent).mat()
#         A.mat().setSizes(M.getSizes())
#         A.mat().setUp()
#         A.mat().assemble()
#         M.copy(result=A.mat())
#         return A


class ExtractedNonlinearSolver:
    """
    Class encapsulating the extra work surrounding a nonlinear solve when
    the problem is posed on an ``ExtractedSpline``.
    """

    def __init__(self, problem, solver):
        """
        ``problem`` is an ``ExtractedNonlinearProblem``, while ``solver``
        is either a ``NewtonSolver`` or a ``PETScSNESSolver``
        that will be used behind the scenes.
        """
        self.problem = problem
        self.solver = solver

    def solve(self):
        """
        This method solves ``self.problem``, using ``self.solver`` and updating
        ``self.problem.solution`` with the solution (in extracted FE
        representation).
        """

        # Need to solve a linear problem for initial guess for IGA DoFs; any
        # way around this?
        tempVec = self.problem.spline.FEtoIGA(self.problem.solution)

        # tempFunc = Function(self.problem.spline.V)
        # tempFunc.assign(self.problem.solution)
        # RHS of problem for initial guess IGA DoFs:
        # MTtemp = self.problem.spline.extractVector(tempFunc.vector(),
        #                                           applyBCs=False)
        # Vector with right dimension for IGA DoFs (content doesn't matter):
        # tempVec = self.problem.spline.extractVector(tempFunc.vector())
        # LHS of problem for initial guess:
        # Mm = as_backend_type(self.problem.spline.M).mat()
        # MTMm = Mm.transposeMatMult(Mm)
        # MTM = PETScMatrix(MTMm)
        # if(self.problem.spline.linearSolver == None):
        #    solve(MTM,tempVec,MTtemp)
        # else:
        #    self.problem.spline.linearSolver.solve(MTM,tempVec,MTtemp)
        self.solver.solve(self.problem, tempVec)

        self.problem.solution.vector()[:] = self.problem.spline.M * tempVec


# class SplineDisplacementExpression(Expression):
#
#    """
#    An expression that can be used to evaluate ``F`` plus an optional
#    displacement at arbitrary points.  To be usable, it must have the
#    following attributes assigned:
#
#    (1) ``self.spline``: an instance of ``ExtractedSpline`` to which the
#    displacement applies.
#
#    (2) ``self.functionList:`` a list of scalar functions in the
#    function space for ``spline``'s control mesh, which act as components of
#    the displacement. If ``functionList`` contains too few entries (including
#    zero entries), the missing entries are assumed to be zero.
#    """
#
#    # needs attributes:
#    # - spline (ExtractedSpline)
#    # - functionList (list of SCALAR Functions)
#
#    def eval_cell(self,values,x,c):
#        phi = []
#        out = array([0.0,])
#        for i in range(0,self.spline.nsd):
#            self.spline.cpFuncs[i].set_allow_extrapolation(True)
#            #phi += [self.cpFuncs[i](Point(x)),]
#            self.spline.cpFuncs[i].eval_cell(out,x,c)
#            phi += [out[0],]
#        self.spline.cpFuncs[self.spline.nsd].set_allow_extrapolation(True)
#        for i in range(0,self.spline.nsd):
#            if(i<len(self.functionList)):
#                self.functionList[i].set_allow_extrapolation(True)
#                self.functionList[i].eval_cell(out,x,c)
#                phi[i] += out[0]
#        #w = self.cpFuncs[self.nsd](Point(x))
#        self.spline.cpFuncs[self.spline.nsd].eval_cell(out,x,c)
#        w = out[0]
#        for i in range(0,self.spline.nsd):
#            phi[i] = phi[i]/w
#        xx = []
#        for i in range(0,self.spline.nsd):
#            if(i<len(x)):
#                xx += [x[i],]
#            else:
#                xx += [0,]
#        for i in range(0,self.spline.nsd):
#            values[i] = phi[i] - xx[i]
#
#    #def value_shape(self):
#    #    return (self.spline.nsd,)


# compose with deformation
# class tIGArxExpression(Expression):
#
#    """
#    A subclass of ``Expression`` which composes its attribute ``self.expr``
#    (also an ``Expression``) with the deformation ``F`` given by its attribute
#    ``self.cpFuncs``, which is a list of ``Function`` objects, specifying the
#    components of ``F``.
#    """
#
#    # using eval_cell allows us to avoid having to search for which cell
#    # x is in; also x need not be in a unique cell, which is nice for
#    # splines that do not have a single coordinate chart
#    def eval_cell(self,values,x,c):
#        phi = []
#        out = array([0.0,])
#        for i in range(0,self.nsd):
#            self.cpFuncs[i].set_allow_extrapolation(True)
#            self.cpFuncs[i].eval_cell(out,x,c)
#            phi += [out[0],]
#        self.cpFuncs[self.nsd].set_allow_extrapolation(True)
#        self.cpFuncs[self.nsd].eval_cell(out,x,c)
#        w = out[0]
#        for i in range(0,self.nsd):
#            phi[i] = phi[i]/w
#        self.expr.eval(values,array(phi))
