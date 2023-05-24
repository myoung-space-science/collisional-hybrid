#ifndef LHS_H
#define LHS_H

#include <petsc.h>
#include "constants.h"

extern PetscErrorCode ComputeIdentityLHS(KSP ksp, Mat J, Mat A, void *_ctx);

extern PetscErrorCode ComputeLaplacianLHS(KSP ksp, Mat J, Mat A, void *_ctx);

extern PetscErrorCode ComputeFullLHS(KSP ksp, Mat J, Mat A, void *_ctx);

extern PetscErrorCode ComputePeriodicStencil(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES], void *ctx);

extern PetscErrorCode ComputeNeumannStencil(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES], void *ctx);

#endif // LHS_H