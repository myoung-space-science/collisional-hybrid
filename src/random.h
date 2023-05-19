#ifndef RANDOM_H
#define RANDOM_H

#include <petsc.h>

extern float ran1(long *idum);

extern PetscErrorCode Ran1(long *idum, PetscReal *result);

extern float ran3(long *idum);

extern PetscErrorCode Ran3(long *idum, PetscReal *result);

extern float gasdev(long *idum);

extern PetscErrorCode Gasdev(long *idum, PetscReal *result);

extern PetscErrorCode Sobseq(PetscInt *n, PetscReal x[]);

#endif // RANDOM_H
