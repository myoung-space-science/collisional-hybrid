petscdir ?= ~/petsc
localbin = ../bin

hybrid: hybrid.o random.o
	-$(CLINKER) -o ${localbin}/$@ $^ $(LDLIBS)

include ${petscdir}/lib/petsc/conf/variables
include ${petscdir}/lib/petsc/conf/rules

