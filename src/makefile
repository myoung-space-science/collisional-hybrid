petscdir ?= ~/petsc

include ${petscdir}/lib/petsc/conf/variables
include ${petscdir}/lib/petsc/conf/rules

localbin = ../bin

hybrid: hybrid.o
	-${CLINKER} -o ${localbin}/hybrid hybrid.o ${PETSC_LIB}
	${RM} hybrid.o

