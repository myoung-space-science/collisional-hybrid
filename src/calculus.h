#ifndef CALCULUS_H
#define CALCULUS_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode dFdx(PetscReal ***F, PetscReal dx, PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, DifferenceType type);
extern PetscErrorCode dFdy(PetscReal ***F, PetscReal dy, PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, DifferenceType type);
extern PetscErrorCode dFdz(PetscReal ***F, PetscReal dz, PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, DifferenceType type);

extern PetscErrorCode d2Fdxx(PetscReal ***F, PetscReal dx, PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, DifferenceType type);
extern PetscErrorCode d2Fdyy(PetscReal ***F, PetscReal dy, PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, DifferenceType type);
extern PetscErrorCode d2Fdzz(PetscReal ***F, PetscReal dz, PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, DifferenceType type);

#endif // CALCULUS_H