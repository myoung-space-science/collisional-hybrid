#ifndef LHS_H
#define LHS_H

#include <petsc.h>
#include "constants.h"

extern const char *LHSTypes[];

typedef enum {
  LHS_IDENTITY,
  LHS_LAPLACIAN,
  LHS_FULL,
} LHSType;

extern PetscErrorCode ComputeIdentityLHS(KSP ksp, Mat J, Mat A, void *ctx);
extern PetscErrorCode ComputeLaplacianLHS(KSP ksp, Mat J, Mat A, void *ctx);
extern PetscErrorCode ComputeFullLHS(KSP ksp, Mat J, Mat A, void *ctx);

typedef PetscErrorCode (*LHSFunc)(KSP ksp, Mat J, Mat A, void *_ctx);

extern PetscErrorCode ComputePeriodicStencil(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES], void *ctx);
extern PetscErrorCode ComputeNeumannStencil(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES], void *ctx);

typedef PetscErrorCode (*StencilFunc)(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES], void *ctx);

#endif // LHS_H