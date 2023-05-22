#ifndef LHS_H
#define LHS_H

#include <petsc.h>

extern PetscErrorCode ComputeIdentityLHS(KSP ksp, Mat J, Mat A, void *_ctx);

extern PetscErrorCode ComputeLaplacianLHS(KSP ksp, Mat J, Mat A, void *_ctx);

extern PetscErrorCode ComputeFullLHS(KSP ksp, Mat J, Mat A, void *_ctx);

#endif // LHS_H