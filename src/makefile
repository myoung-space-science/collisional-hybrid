petscdir ?= ~/petsc
localbin = ../bin

hybrid: hybrid.o \
		random.o \
		lhs.o \
		rhs.o \
		output.o \
		parameters.o \
		initialize.o \
		distributions.o \
		potential.o \
		particles.o
	-$(CLINKER) -o ${localbin}/$@ $^ $(LDLIBS)

include ${petscdir}/lib/petsc/conf/variables
include ${petscdir}/lib/petsc/conf/rules

