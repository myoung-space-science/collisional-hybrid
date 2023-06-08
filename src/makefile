petscdir ?= ~/petsc
localbin = ../bin

default: hybrid

hybrid: hybrid.o \
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
		fileio.o \
		lhs.o \
		parameters.o \
		potential.o \
		rhs.o \
		setup.o
	-$(CLINKER) -o ${localbin}/$@ $^ $(LDLIBS) $(SLEPC_EPS_LIB)

include ${petscdir}/lib/petsc/conf/variables
include ${petscdir}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

