localbin = ../bin

default: hybrid

hybrid: hybrid.o \
		calculus.o \
		distributions.o \
		fileio.o \
		lhs.o \
		parameters.o \
		particles.o \
		potential.o \
		random.o \
		rhs.o \
		setup.o \
		vectors.o
	-$(CLINKER) -o ${localbin}/$@ $^ $(LDLIBS)

solver: solver.o \
		calculus.o \
		distributions.o \
		fileio.o \
		lhs.o \
		parameters.o \
		particles.o \
		potential.o \
		random.o \
		rhs.o \
		setup.o \
		vectors.o
	-$(CLINKER) -o ${localbin}/$@ $^ $(LDLIBS) $(SLEPC_EPS_LIB)

include ${PETSC_DIR}/lib/petsc/conf/variables
# The following line will implicitly include PETSc rules. If we decide to remove
# it, we will need to explicitly include `${PETSC_DIR}/lib/petsc/conf/rules`.
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

