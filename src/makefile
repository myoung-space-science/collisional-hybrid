petscdir ?= ~/petsc
localbin = ../bin

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
	-$(CLINKER) -o ${localbin}/$@ $^ $(LDLIBS)

include ${petscdir}/lib/petsc/conf/variables
include ${petscdir}/lib/petsc/conf/rules

