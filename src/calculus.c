#include <petsc.h>
#include "hybrid.h"


/* Compute f(i,j,k) = dF/dx|(i,j,k) with a second-order difference scheme. */
PetscErrorCode dFdx(PetscReal ***F, PetscReal dx, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType type)
{
  PetscReal v;

  PetscFunctionBeginUser;

  switch (type) {
    case CENTERED:
      v = F[k][j][i+1] - F[k][j][i-1];
      break;
    case FORWARD:
      v = -1.0*F[k][j][i+2] + 4.0*F[k][j][i+1] - 3.0*F[k][j][i];
      break;
    case BACKWARD:
      v = +3.0*F[k][j][i] - 4.0*F[k][j][i-1] + 1.0*F[k][j][i-2];
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }
  if (dx > 0.0) {
    *f = v / (2.0*dx);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute f(i,j,k) = dF/dy|(i,j,k) with a second-order difference scheme. */
PetscErrorCode dFdy(PetscReal ***F, PetscReal dy, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType type)
{
  PetscReal v;

  PetscFunctionBeginUser;

  switch (type) {
    case CENTERED:
      v = F[k][j+1][i] - F[k][j-1][i];
      break;
    case FORWARD:
      v = -1.0*F[k][j+2][i] + 4.0*F[k][j+1][i] - 3.0*F[k][j][i];
      break;
    case BACKWARD:
      v = +3.0*F[k][j][i] - 4.0*F[k][j-1][i] + 1.0*F[k][j-2][i];
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }
  if (dy > 0.0) {
    *f = v / (2.0*dy);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute f(i,j,k) = dF/dz|(i,j,k) with a second-order difference scheme. */
PetscErrorCode dFdz(PetscReal ***F, PetscReal dz, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType type)
{
  PetscReal v;

  PetscFunctionBeginUser;

  switch (type) {
    case CENTERED:
      v = F[k+1][j][i] - F[k-1][j][i];
      break;
    case FORWARD:
      v = -1.0*F[k+2][j][i] + 4.0*F[k+1][j][i] - 3.0*F[k][j][i];
      break;
    case BACKWARD:
      v = +3.0*F[k][j][i] - 4.0*F[k-1][j][i] + 1.0*F[k-2][j][i];
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }
  if (dz > 0.0) {
    *f = v / (2.0*dz);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute f(i,j,k) = d^2F/dx^2|(i,j,k) with a second-order difference scheme. */
PetscErrorCode d2Fdxx(PetscReal ***F, PetscReal dx, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType type)
{
  PetscReal v;

  PetscFunctionBeginUser;

  switch (type) {
    case CENTERED:
      v = F[k][j][i+1] - 2.0*F[k][j][i] + F[k][j][i-1];
      break;
    case FORWARD:
      v = -1.0*F[k][j][i+3] + 4.0*F[k][j][i+2] - 5.0*F[k][j][i+1] + 2.0*F[k][j][i];
      break;
    case BACKWARD:
      v = -1.0*F[k][j][i-3] + 4.0*F[k][j][i-2] - 5.0*F[k][j][i-1] + 2.0*F[k][j][i];
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }
  if (dx > 0.0) {
    *f = v / (dx*dx);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute f(i,j,k) = d^2F/dy^2|(i,j,k) with a second-order difference scheme. */
PetscErrorCode d2Fdyy(PetscReal ***F, PetscReal dy, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType type)
{
  PetscReal v;

  PetscFunctionBeginUser;

  switch (type) {
    case CENTERED:
      v = F[k][j+1][i] - 2.0*F[k][j][i] + F[k][j-1][i];
      break;
    case FORWARD:
      v = -1.0*F[k][j+3][i] + 4.0*F[k][j+2][i] - 5.0*F[k][j+1][i] + 2.0*F[k][j][i];
      break;
    case BACKWARD:
      v = -1.0*F[k][j-3][i] + 4.0*F[k][j-2][i] - 5.0*F[k][j-1][i] + 2.0*F[k][j][i];
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }
  if (dy > 0.0) {
    *f = v / (dy*dy);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute f(i,j,k) = d^2F/dz^2|(i,j,k) with a second-order difference scheme. */
PetscErrorCode d2Fdzz(PetscReal ***F, PetscReal dz, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType type)
{
  PetscReal v;

  PetscFunctionBeginUser;

  switch (type) {
    case CENTERED:
      v = F[k+1][j][i] - 2.0*F[k][j][i] + F[k-1][j][i];
      break;
    case FORWARD:
      v = -1.0*F[k+3][j][i] + 4.0*F[k+2][j][i] - 5.0*F[k+1][j][i] + 2.0*F[k][j][i];
      break;
    case BACKWARD:
      v = -1.0*F[k-3][j][i] + 4.0*F[k-2][j][i] - 5.0*F[k-1][j][i] + 2.0*F[k][j][i];
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }
  if (dz > 0.0) {
    *f = v / (dz*dz);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


