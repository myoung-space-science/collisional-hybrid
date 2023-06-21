#include <petsc.h>
#include "hybrid.h"


/* Compute dF/dx to second order.

  Approximate f(i,j,k) = dF/dx at (i,j,k) using second-order centered, forward,
  or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dx [in]: The cell width. Passing dx <= 0.0 will cause this function to ignore
    this parameter and return the differential quantity dF. Doing so may be
    useful in cases when calling code prefers to scale the value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - type [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
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


/* Compute dF/dy to second order.

  Approximate f(i,j,k) = dF/dy at (i,j,k) using second-order centered, forward,
  or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dy [in]: The cell width. Passing dy <= 0.0 will cause this function to ignore
    this parameter and return the differential quantity dF. Doing so may be
    useful in cases when calling code prefers to scale the value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - type [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
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


/* Compute dF/dz to second order.

  Approximate f(i,j,k) = dF/dz at (i,j,k) using second-order centered, forward,
  or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dz [in]: The cell width. Passing dz <= 0.0 will cause this function to ignore
    this parameter and return the differential quantity dF. Doing so may be
    useful in cases when calling code prefers to scale the value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - type [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
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


/* Compute d(dF/dx)/dx to second order.

  Approximate f(i,j,k) = d(dF/dx)/dx at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dx [in]: The cell width. Passing dx <= 0.0 will cause this function to
    ignore this parameter and return the differential quantity dF. Doing so may
    be useful in cases when calling code prefers to scale the value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - type [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
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


/* Compute d(dF/dy)/dy to second order.

  Approximate f(i,j,k) = d(dF/dy)/dy at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dy [in]: The cell width. Passing dy <= 0.0 will cause this function to
    ignore this parameter and return the differential quantity dF. Doing so may
    be useful in cases when calling code prefers to scale the value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - type [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
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


/* Compute d(dF/dz)/dz to second order.

  Approximate f(i,j,k) = d(dF/dz)/dz at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dz [in]: The cell width. Passing dz <= 0.0 will cause this function to
    ignore this parameter and return the differential quantity dF. Doing so may
    be useful in cases when calling code prefers to scale the value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - type [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
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


/* Compute d(dF/dx)/dy to second order.

  Approximate f(i,j,k) = d(dF/dx)/dy at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dx, dy [in]: The cell widths. Passing dx <= 0.0 or dy <= 0.0 will cause this
    function to ignore this parameter and return the differential quantity dF.
    Doing so may be useful in cases when calling code prefers to scale the
    value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - xtype [in]: One of CENTERED, FOWARD, or BACKWARD.
  - ytype [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
PetscErrorCode d2Fdxy(PetscReal ***F, PetscReal dx, PetscReal dy, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType xtype, DifferenceType ytype)
{
  PetscReal v;

  PetscFunctionBeginUser;

         if ((xtype == CENTERED) && (ytype == CENTERED)) {
    v = F[k][j+1][i+1] - F[k][j+1][i-1] - F[k][j-1][i+1] + F[k][j-1][i-1];
  } else if ((xtype == CENTERED) && (ytype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "CENTERED+FORWARD not implemented for d(dF/dx)/dy.\n");
  } else if ((xtype == CENTERED) && (ytype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "CENTERED+BACKWARD not implemented for d(dF/dx)/dy.\n");
  } else if ((xtype == FORWARD)  && (ytype == CENTERED)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+CENTERED not implemented for d(dF/dx)/dy.\n");
  } else if ((xtype == FORWARD)  && (ytype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+FORWARD not implemented for d(dF/dx)/dy.\n");
  } else if ((xtype == FORWARD)  && (ytype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+BACKWARD not implemented for d(dF/dx)/dy.\n");
  } else if ((xtype == BACKWARD) && (ytype == CENTERED)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+CENTERED not implemented for d(dF/dx)/dy.\n");
  } else if ((xtype == BACKWARD) && (ytype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+FORWARD not implemented for d(dF/dx)/dy.\n");
  } else if ((xtype == BACKWARD) && (ytype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+BACKWARD not implemented for d(dF/dx)/dy.\n");
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }

  if ((dx > 0.0) && (dy > 0.0)) {
    *f = v / (4.0*dx*dy);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute d(dF/dy)/dx to second order.

  Approximate f(i,j,k) = d(dF/dy)/dx at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dy, dx [in]: The cell widths. Passing dy <= 0.0 or dx <= 0.0 will cause this
    function to ignore this parameter and return the differential quantity dF.
    Doing so may be useful in cases when calling code prefers to scale the
    value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - ytype [in]: One of CENTERED, FOWARD, or BACKWARD.
  - xtype [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
PetscErrorCode d2Fdyx(PetscReal ***F, PetscReal dy, PetscReal dx, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType ytype, DifferenceType xtype)
{
  PetscFunctionBeginUser;
  PetscCall(d2Fdxy(F, dx, dy, i, j, k, f, xtype, ytype));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute d(dF/dx)/dz to second order.

  Approximate f(i,j,k) = d(dF/dx)/dz at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dx, dz [in]: The cell widths. Passing dx <= 0.0 or dz <= 0.0 will cause this
    function to ignore this parameter and return the differential quantity dF.
    Doing so may be useful in cases when calling code prefers to scale the
    value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - xtype [in]: One of CENTERED, FOWARD, or BACKWARD.
  - ztype [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
PetscErrorCode d2Fdxz(PetscReal ***F, PetscReal dx, PetscReal dz, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType xtype, DifferenceType ztype)
{
  PetscReal v;

  PetscFunctionBeginUser;

         if ((xtype == CENTERED) && (ztype == CENTERED)) {
    v = F[k+1][j][i+1] - F[k+1][j][i-1] - F[k-1][j][i+1] + F[k-1][j][i-1];
  } else if ((xtype == CENTERED) && (ztype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "CENTERED+FORWARD not implemented for d(dF/dx)/dz.\n");
  } else if ((xtype == CENTERED) && (ztype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "CENTERED+BACKWARD not implemented for d(dF/dx)/dz.\n");
  } else if ((xtype == FORWARD)  && (ztype == CENTERED)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+CENTERED not implemented for d(dF/dx)/dz.\n");
  } else if ((xtype == FORWARD)  && (ztype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+FORWARD not implemented for d(dF/dx)/dz.\n");
  } else if ((xtype == FORWARD)  && (ztype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+BACKWARD not implemented for d(dF/dx)/dz.\n");
  } else if ((xtype == BACKWARD) && (ztype == CENTERED)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+CENTERED not implemented for d(dF/dx)/dz.\n");
  } else if ((xtype == BACKWARD) && (ztype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+FORWARD not implemented for d(dF/dx)/dz.\n");
  } else if ((xtype == BACKWARD) && (ztype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+BACKWARD not implemented for d(dF/dx)/dz.\n");
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }

  if ((dx > 0.0) && (dz > 0.0)) {
    *f = v / (4.0*dx*dz);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute d(dF/dz)/dx to second order.

  Approximate f(i,j,k) = d(dF/dz)/dx at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dz, dx [in]: The cell widths. Passing dz <= 0.0 or dx <= 0.0 will cause this
    function to ignore this parameter and return the differential quantity dF.
    Doing so may be useful in cases when calling code prefers to scale the
    value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - ztype [in]: One of CENTERED, FOWARD, or BACKWARD.
  - xtype [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
PetscErrorCode d2Fdzx(PetscReal ***F, PetscReal dz, PetscReal dx, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType ztype, DifferenceType xtype)
{
  PetscFunctionBeginUser;
  PetscCall(d2Fdxz(F, dx, dz, i, j, k, f, xtype, ztype));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute d(dF/dy)/dz to second order.

  Approximate f(i,j,k) = d(dF/dy)/dz at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dy, dz [in]: The cell widths. Passing dy <= 0.0 or dz <= 0.0 will cause this
    function to ignore this parameter and return the differential quantity dF.
    Doing so may be useful in cases when calling code prefers to scale the
    value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - ytype [in]: One of CENTERED, FOWARD, or BACKWARD.
  - ztype [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
PetscErrorCode d2Fdyz(PetscReal ***F, PetscReal dy, PetscReal dz, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType ytype, DifferenceType ztype)
{
  PetscReal v;

  PetscFunctionBeginUser;

         if ((ytype == CENTERED) && (ztype == CENTERED)) {
    v = F[k+1][j][i+1] - F[k+1][j][i-1] - F[k-1][j][i+1] + F[k-1][j][i-1];
  } else if ((ytype == CENTERED) && (ztype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "CENTERED+FORWARD not implemented for d(dF/dy)/dz.\n");
  } else if ((ytype == CENTERED) && (ztype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "CENTERED+BACKWARD not implemented for d(dF/dy)/dz.\n");
  } else if ((ytype == FORWARD)  && (ztype == CENTERED)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+CENTERED not implemented for d(dF/dy)/dz.\n");
  } else if ((ytype == FORWARD)  && (ztype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+FORWARD not implemented for d(dF/dy)/dz.\n");
  } else if ((ytype == FORWARD)  && (ztype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FORWARD+BACKWARD not implemented for d(dF/dy)/dz.\n");
  } else if ((ytype == BACKWARD) && (ztype == CENTERED)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+CENTERED not implemented for d(dF/dy)/dz.\n");
  } else if ((ytype == BACKWARD) && (ztype == FORWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+FORWARD not implemented for d(dF/dy)/dz.\n");
  } else if ((ytype == BACKWARD) && (ztype == BACKWARD)) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "BACKWARD+BACKWARD not implemented for d(dF/dy)/dz.\n");
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Accepted difference types are FORWARD, BACKWARD, and CENTERED.\n");
  }

  if ((dy > 0.0) && (dz > 0.0)) {
    *f = v / (4.0*dy*dz);
  } else {
    *f = v;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute d(dF/dz)/dy to second order.

  Approximate f(i,j,k) = d(dF/dz)/dy at (i,j,k) using second-order centered,
  forward, or backward differencing scheme.

  - F [in]: The antiderivative function.
  - dz, dy [in]: The cell widths. Passing dz <= 0.0 or dy <= 0.0 will cause this
    function to ignore this parameter and return the differential quantity dF.
    Doing so may be useful in cases when calling code prefers to scale the
    value.
  - i [in]: The x-axis index.
  - j [in]: The y-axis index.
  - k [in]: The z-axis index.
  - f [out]: A pointer to the scalar result.
  - ztype [in]: One of CENTERED, FOWARD, or BACKWARD.
  - ytype [in]: One of CENTERED, FOWARD, or BACKWARD.

*/
PetscErrorCode d2Fdzy(PetscReal ***F, PetscReal dz, PetscReal dy, PetscInt i, PetscInt j, PetscInt k, PetscReal *f, DifferenceType ztype, DifferenceType ytype)
{
  PetscFunctionBeginUser;
  PetscCall(d2Fdyz(F, dy, dz, i, j, k, f, ytype, ztype));
  PetscFunctionReturn(PETSC_SUCCESS);
}


