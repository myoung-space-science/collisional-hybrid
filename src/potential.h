#ifndef POTENTIAL_H
#define POTENTIAL_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode ComputePotential(KSP ksp, Context *ctx);

extern PetscErrorCode ComputeInitialPhi(KSP ksp, Vec phi, void *ctx);

extern PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx);

extern PetscErrorCode ComputeLHS(KSP ksp, Mat J, Mat A, void *ctx);

#endif // POTENTIAL_H