#include <petsc.h>
#include "hybrid.h"


/* Compute the electrostatic potential and store the result in the context. */
PetscErrorCode ComputePotential(KSP ksp, Context *ctx)
{
  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(KSPGetSolution(ksp, &ctx->phi));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Set the initial guess for the electrostatic potential. */
PetscErrorCode ComputeInitialPhi(KSP ksp, Vec phi, void *ctx)
{
  // Note that `KSPSetComputeInitialGuess` requires this function signature.

  PetscFunctionBeginUser;

  PetscCall(VecSet(phi, 0.0));

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute the forcing vector for the electrostatic-potential equation. */
PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  Context      *user=(Context *)ctx;

  PetscFunctionBeginUser;

  PetscCall(user->rhsFunc(ksp, b, ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute the operator matrix for the electrostatic-potential equation. */
PetscErrorCode ComputeLHS(KSP ksp, Mat J, Mat A, void *ctx)
{
  Context      *user=(Context *)ctx;

  PetscFunctionBeginUser;

  PetscCall(user->lhsFunc(ksp, J, A, ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}


