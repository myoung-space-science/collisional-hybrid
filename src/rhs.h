#ifndef RHS_H
#define RHS_H

#include <petsc.h>

extern PetscErrorCode ComputeConstantRHS(KSP ksp, Vec b, void *_ctx);

extern PetscErrorCode ComputeSinusoidalRHS(KSP ksp, Vec b, void *_ctx);

extern PetscErrorCode ComputeFullRHS(KSP ksp, Vec b, void *_ctx);


#endif // RHS_H