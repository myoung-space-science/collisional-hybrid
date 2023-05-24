#include <petsc.h>
#include "hybrid.h"


/* Compute a vector of central differences from F.

This function was designed as the first step in computing the gradient of the
scalar function F(x, y, z) at (x0, y0, z0). It computes the numerator of each
finite-difference term using 2nd-order centered, forward, and backward
approximations. It assumes that F contains the appropriate ghost nodes.
*/
PetscErrorCode DifferenceVector(PetscReal ***F, PetscReal x0, PetscReal y0, PetscReal z0, Grid grid, PetscReal f[NDIM])
{
  PetscInt    Nx=grid.N.x, Ny=grid.N.y, Nz=grid.N.z;
  PetscInt    ixl, ixh, iyl, iyh, izl, izh;
  PetscReal   wxh, wyh, wzh;
  PetscReal   hhh, lhh, hlh, llh, hhl, lhl, hll, lll;
  PetscReal   whh, whl, wlh, wll, Ewh, Ewl;

  PetscFunctionBeginUser;

  // Compute the x-dimension neighbors and corresponding weights.
  ixl = (PetscInt)x0;
  ixh = ixl+1;
  wxh = x0 - (PetscReal)ixl;
  // Compute the y-dimension neighbors and corresponding weights.
  iyl = (PetscInt)y0;
  iyh = iyl+1;
  wyh = y0 - (PetscReal)iyl;
  // Compute the z-dimension neighbors and corresponding weights.
  izl = (PetscInt)z0;
  izh = izl+1;
  wzh = z0 - (PetscReal)izl;
  // Compute the central difference in x at each grid point.
  if (ixl >= 0) {
    // 2nd-order central difference at ixl
    hhl = F[izh][iyh][ixl+1] - F[izh][iyh][ixl-1];
    lhl = F[izl][iyh][ixl+1] - F[izl][iyh][ixl-1];
    hll = F[izh][iyl][ixl+1] - F[izh][iyl][ixl-1];
    lll = F[izl][iyl][ixl+1] - F[izl][iyl][ixl-1];
  } else {
    // 2nd-order forward difference at ixl
    hhl = -1.0*F[izh][iyh][ixl+2] + 4.0*F[izh][iyh][ixl+1] - 3.0*F[izh][iyh][ixl];
    lhl = -1.0*F[izl][iyh][ixl+2] + 4.0*F[izl][iyh][ixl+1] - 3.0*F[izl][iyh][ixl];
    hll = -1.0*F[izh][iyl][ixl+2] + 4.0*F[izh][iyl][ixl+1] - 3.0*F[izh][iyl][ixl];
    lll = -1.0*F[izl][iyl][ixl+2] + 4.0*F[izl][iyl][ixl+1] - 3.0*F[izl][iyl][ixl];
  }
  if (ixh < Nx) {
    // 2nd-order central difference at ixh
    hhh = F[izh][iyh][ixh+1] - F[izh][iyh][ixh-1];
    lhh = F[izl][iyh][ixh+1] - F[izl][iyh][ixh-1];
    hlh = F[izh][iyl][ixh+1] - F[izh][iyl][ixh-1];
    llh = F[izl][iyl][ixh+1] - F[izl][iyl][ixh-1];
  } else {
    // 2nd-order backward difference at ixh
    hhh = +3.0*F[izh][iyh][ixh] - 4.0*F[izh][iyh][ixh-1] + 1.0*F[izh][iyh][ixh-2];
    lhh = +3.0*F[izl][iyh][ixh] - 4.0*F[izl][iyh][ixh-1] + 1.0*F[izl][iyh][ixh-2];
    hlh = +3.0*F[izh][iyl][ixh] - 4.0*F[izh][iyl][ixh-1] + 1.0*F[izh][iyl][ixh-2];
    llh = +3.0*F[izl][iyl][ixh] - 4.0*F[izl][iyl][ixh-1] + 1.0*F[izl][iyl][ixh-2];
  }
  whh = hlh + wyh*(hhh - hlh);
  whl = hll + wyh*(hhl - hll);
  wlh = llh + wyh*(lhh - llh);
  wll = lll + wyh*(lhl - lll);
  Ewh = wlh + wxh*(whh - wlh);
  Ewl = wll + wxh*(whl - wll);
  f[0] = Ewl + wzh*(Ewh - Ewh);
  // Compute the central difference in y at each grid point.
  if (iyl >= 0) {
    // 2nd-order central difference at iyl
    hlh = F[izh][iyl+1][ixh] - F[izh][iyl-1][ixh];
    llh = F[izl][iyl+1][ixh] - F[izl][iyl-1][ixh];
    hll = F[izh][iyl+1][ixl] - F[izh][iyl-1][ixl];
    lll = F[izl][iyl+1][ixl] - F[izl][iyl-1][ixl];
  } else {
    // 2nd-order forward difference at iyl
    hlh = -1.0*F[izh][iyl+2][ixh] + 4.0*F[izh][iyl+1][ixh] - 3.0*F[izh][iyl][ixh];
    llh = -1.0*F[izl][iyl+2][ixh] + 4.0*F[izl][iyl+1][ixh] - 3.0*F[izl][iyl][ixh];
    hll = -1.0*F[izh][iyl+2][ixl] + 4.0*F[izh][iyl+1][ixl] - 3.0*F[izh][iyl][ixl];
    lll = -1.0*F[izl][iyl+2][ixl] + 4.0*F[izl][iyl+1][ixl] - 3.0*F[izl][iyl][ixl];
  }
  if (iyh < Ny) {
    // 2nd-order central difference at iyh
    hhh = F[izh][iyh+1][ixh] - F[izh][iyh-1][ixh];
    lhh = F[izl][iyh+1][ixh] - F[izl][iyh-1][ixh];
    hhl = F[izh][iyh+1][ixl] - F[izh][iyh-1][ixl];
    lhl = F[izl][iyh+1][ixl] - F[izl][iyh-1][ixl];
  } else {
    // 2nd-order backward difference at iyh
    hhh = +3.0*F[izh][iyh][ixh] - 4.0*F[izh][iyh-1][ixh] + 1.0*F[izh][iyh-2][ixh];
    lhh = +3.0*F[izl][iyh][ixh] - 4.0*F[izl][iyh-1][ixh] + 1.0*F[izl][iyh-2][ixh];
    hhl = +3.0*F[izh][iyh][ixl] - 4.0*F[izh][iyh-1][ixl] + 1.0*F[izh][iyh-2][ixl];
    lhl = +3.0*F[izl][iyh][ixl] - 4.0*F[izl][iyh-1][ixl] + 1.0*F[izl][iyh-2][ixl];
  }
  whh = hlh + wyh*(hhh - hlh);
  whl = hll + wyh*(hhl - hll);
  wlh = llh + wyh*(lhh - llh);
  wll = lll + wyh*(lhl - lll);
  Ewh = wlh + wxh*(whh - wlh);
  Ewl = wll + wxh*(whl - wll);
  f[1] = Ewl + wzh*(Ewh - Ewh);
  // Compute the central difference in z at each grid point.
  if (izl >= 0) {
    // 2nd-order central difference at izl
    lhh = F[izl+1][iyh][ixh] - F[izl-1][iyh][ixh];
    llh = F[izl+1][iyl][ixh] - F[izl-1][iyl][ixh];
    lhl = F[izl+1][iyh][ixl] - F[izl-1][iyh][ixl];
    lll = F[izl+1][iyl][ixl] - F[izl-1][iyl][ixl];
  } else {
    // 2nd-order forward difference at izl
    lhh = -1.0*F[izl+2][iyh][ixh] + 4.0*F[izl+1][iyh][ixh] - 3.0*F[izl][iyh][ixh];
    llh = -1.0*F[izl+2][iyl][ixh] + 4.0*F[izl+1][iyl][ixh] - 3.0*F[izl][iyl][ixh];
    lhl = -1.0*F[izl+2][iyh][ixl] + 4.0*F[izl+1][iyh][ixl] - 3.0*F[izl][iyh][ixl];
    lll = -1.0*F[izl+2][iyl][ixl] + 4.0*F[izl+1][iyl][ixl] - 3.0*F[izl][iyl][ixl];
  }
  if (izh < Nz) {
    // 2nd-order central difference at izh
    hhh = F[izh+1][iyh][ixh] - F[izh-1][iyh][ixh];
    hlh = F[izh+1][iyl][ixh] - F[izh-1][iyl][ixh];
    hhl = F[izh+1][iyh][ixl] - F[izh-1][iyh][ixl];
    hll = F[izh+1][iyl][ixl] - F[izh-1][iyl][ixl];
  } else {
    // 2nd-order backward difference at izh
    hhh = +3.0*F[izh][iyh][ixh] - 4.0*F[izh-1][iyh][ixh] + 1.0*F[izh-2][iyh][ixh];
    hlh = +3.0*F[izh][iyl][ixh] - 4.0*F[izh-1][iyl][ixh] + 1.0*F[izh-2][iyl][ixh];
    hhl = +3.0*F[izh][iyh][ixl] - 4.0*F[izh-1][iyh][ixl] + 1.0*F[izh-2][iyh][ixl];
    hll = +3.0*F[izh][iyl][ixl] - 4.0*F[izh-1][iyl][ixl] + 1.0*F[izh-2][iyl][ixl];
  }
  whh = hlh + wyh*(hhh - hlh);
  whl = hll + wyh*(hhl - hll);
  wlh = llh + wyh*(lhh - llh);
  wll = lll + wyh*(lhl - lll);
  Ewh = wlh + wxh*(whh - wlh);
  Ewl = wll + wxh*(whl - wll);
  f[2] = Ewl + wzh*(Ewh - Ewh);

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute the dot product given by $\vec{c} = \vec{a} \cdot \vec{b}$. */
PetscErrorCode DotProduct(PetscReal a[NDIM], PetscReal b[NDIM], PetscReal *c)
{
  PetscFunctionBeginUser;

  *c = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute the cross product given by $\vec{c} = \vec{a} \times \vec{b}$. */
PetscErrorCode CrossProduct(PetscReal a[NDIM], PetscReal b[NDIM], PetscReal c[NDIM])
{
  PetscFunctionBeginUser;

  c[0] = a[1]*b[2] - b[2]*a[1];
  c[1] = a[2]*b[0] - b[0]*a[2];
  c[2] = a[0]*b[1] - b[1]*a[0];

  PetscFunctionReturn(PETSC_SUCCESS);
}


